import pathway as pw
from pathway.xpacks.llm import vector_store
import time
import json
import logging
import threading
from typing import List, Optional, Dict, Tuple, Any
import requests
import torch
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
import numpy as np
import io
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
import re
from PDFparserFITZ import DoraemonPDFParser
from Scibert_embeddings import DoraemonProcessor

@dataclass
class ProcessedSection:
    heading: str
    content: str
    vector: np.ndarray
    keywords: List[Tuple[str, float]]
    metadata: Dict[str, Any]

class EnhancedPDFVectorStore:
    def __init__(
        self,
        credentials_file: str,
        folder_id: str,
        host: str = "localhost",
        port: int = 8080,
        debug: bool = True,
        max_workers: int = 4,
        min_text_length: int = 50,
        chunk_size: int = 512  
    ):
        self.credentials_file = credentials_file
        self.folder_id = folder_id
        self.host = host
        self.port = port
        self.debug = debug
        self.max_workers = max_workers
        self.min_text_length = min_text_length
        self.chunk_size = chunk_size
        self.server = None
        self.server_thread = None
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        logging.basicConfig(
            level=logging.DEBUG if debug else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self._init_processors()
        self._init_google_drive()
        
    def _validate_text(self, text: str) -> bool:
        """Validate if the text meets minimum requirements for processing."""
        if not isinstance(text, str):
            self.logger.warning(f"Invalid text type: {type(text)}")
            return False
            
        cleaned_text = text.strip()
        if len(cleaned_text) < self.min_text_length:
            self.logger.warning(f"Text too short: {len(cleaned_text)} chars")
            return False
            
        words = re.findall(r'\b[a-zA-Z]+\b', cleaned_text)
        if len(words) < 5:
            self.logger.warning(f"Too few words found: {len(words)}")
            return False
            
        return True

    def _init_processors(self):
        """Initialize and verify the PDF parser and SciBERT processor."""
        try:
            self.pdf_parser = DoraemonPDFParser()
            self.scibert_processor = DoraemonProcessor()
            
            test_text = "This is a test document for verification."
            test_result = self.scibert_processor.process_document(test_text)
            if test_result is None:
                raise RuntimeError("SciBERT processor failed initialization test")
                
            self.logger.info("Successfully initialized and verified processors")
        except Exception as e:
            self.logger.error(f"Failed to initialize processors: {str(e)}")
            raise RuntimeError("Component initialization failed") from e

    def _init_google_drive(self):
        """Initialize and verify Google Drive connection."""
        try:
            self.credentials = service_account.Credentials.from_service_account_file(
                self.credentials_file,
                scopes=["https://www.googleapis.com/auth/drive.readonly"]
            )
            self.drive_service = build('drive', 'v3', credentials=self.credentials)
            
            try:
                self.drive_service.files().list(
                    q=f"'{self.folder_id}' in parents",
                    pageSize=1
                ).execute()
            except Exception as e:
                raise RuntimeError(f"Cannot access folder {self.folder_id}: {str(e)}")
                
            self.logger.info("Successfully connected to Google Drive and verified folder access")
        except Exception as e:
            self.logger.error(f"Failed to initialize Google Drive: {str(e)}")
            raise

    def process_section(self, heading: str, content: str) -> Optional[ProcessedSection]:
        """Process a single section of text using SciBERT."""
        try:
            if not self._validate_text(content):
                self.logger.warning(f"Skipping invalid section: {heading}")
                return None
                
            self.logger.debug(f"Processing section: {heading[:50]}...")
            processed_result = self.scibert_processor.process_document(content)
            
            if processed_result is None:
                self.logger.warning(f"SciBERT processing failed for section: {heading}")
                return None
            combined_features, _, _, keywords = processed_result
            
            if combined_features is None or len(combined_features) == 0:
                self.logger.warning(f"No features generated for section: {heading}")
                return None
                
            self.logger.debug(f"Successfully processed section: {heading[:50]}")
            return ProcessedSection(
                heading=heading,
                content=content,
                vector=combined_features.cpu().numpy(),
                keywords=keywords,
                metadata={
                    'section_length': len(content),
                    'word_count': len(content.split()),
                    'processed_at': time.time()
                }
            )
        except Exception as e:
            self.logger.error(f"Failed to process section '{heading}': {str(e)}")
            return None

    def process_pdf_content(self, pdf_content: bytes) -> List[Dict]:
        """Process the content of a PDF file and return processed sections."""
        try:
            temp_path = Path("temp.pdf")
            temp_path.write_bytes(pdf_content)
            
            sections = self.pdf_parser.parse_pdf(str(temp_path))
            temp_path.unlink()
            
            if not sections:
                self.logger.warning("No sections extracted from PDF")
                return []
                
            processed_sections = []
            for heading, content in sections.items():
                if self._validate_text(content):
                    result = self.process_section(heading, content)
                    if result:
                        processed_sections.append({
                            'heading': result.heading,
                            'content': result.content,
                            'vector': result.vector,
                            'keywords': result.keywords,
                            'metadata': result.metadata
                        })
            
            if not processed_sections:
                self.logger.warning("No valid sections processed from PDF")
                
            return processed_sections
            
        except Exception as e:
            self.logger.error(f"Failed to process PDF content: {str(e)}")
            return []

    def setup_vector_store(self):
        """Set up the vector store with processed PDF documents."""
        try:
            processed_documents = self._process_initial_pdfs()
            if not processed_documents:
                raise RuntimeError("No valid sections found in initial PDFs")
            
            pdfs = pw.io.gdrive.read(
                object_id=self.folder_id,
                service_user_credentials_file=self.credentials_file,
                mode="streaming",
                file_name_pattern="*.pdf",
                with_metadata=True
            )
            
            self.server = vector_store.VectorStoreServer(
                pdfs,
                embedder=lambda text: self.scibert_processor.process_document(text)[0].cpu().numpy(),
                parser=self._pdf_parser_wrapper
            )
            
            self.logger.info("Vector store setup completed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to setup vector store: {str(e)}")
            raise
    def _process_initial_pdfs(self, sample_size: int = 5) -> List[Dict]:
        """Process a sample of PDFs to verify setup."""
        available_files = self.list_pdf_files()
        if not available_files:
            raise RuntimeError("No PDF files found in the specified folder")
        
        self.logger.info(f"Processing {len(available_files)} PDF files")
        processed_documents = []
        
        for file in available_files[:sample_size]:
            self.logger.info(f"Processing file: {file['name']}")
            pdf_content = self.download_pdf(file['id'])
            if pdf_content:
                sections = self.process_pdf_content(pdf_content)
                if sections:
                    processed_documents.extend(sections)
                    
        return processed_documents

    def _pdf_parser_wrapper(self, pdf_data: bytes, metadata: Dict) -> List[Tuple[str, Dict]]:
        """Wrapper function for PDF parsing to match vector store requirements."""
        sections = self.process_pdf_content(pdf_data)
        if not sections:
            return []
            
        results = []
        for section in sections:
            vector_data = (
                section['content'],
                {
                    'heading': section['heading'],
                    'keywords': [k[0] for k in section['keywords']],
                    'file_name': metadata.get('name', ''),
                    'file_id': metadata.get('id', ''),
                    'vector': section['vector'].tolist(),
                    **section['metadata']
                }
            )
            results.append(vector_data)
        
        return results

    def start(self, timeout: int = 30):
        try:
            self.setup_vector_store()
            
            self.server_thread = threading.Thread(
                target=self.server.run_server,
                kwargs={
                    "host": self.host,
                    "port": self.port,
                    "threaded": True
                }
            )
            self.server_thread.daemon = True
            self.server_thread.start()
            
            if not self.wait_for_server(timeout):
                raise RuntimeError(f"Server failed to start within {timeout} seconds")
                
            self.logger.info(f"Server started successfully on {self.host}:{self.port}")
            return self
            
        except Exception as e:
            self.logger.error(f"Failed to start server: {str(e)}")
            raise

    def get_client(self):
        """Get a client instance for the vector store."""
        return vector_store.Client(host=self.host, port=self.port)

    def list_pdf_files(self) -> List[Dict]:
        """List all PDF files in the specified Google Drive folder."""
        try:
            results = []
            page_token = None
            while True:
                response = self.drive_service.files().list(
                    q=f"'{self.folder_id}' in parents and mimeType='application/pdf'",
                    spaces='drive',
                    fields='nextPageToken, files(id, name)',
                    pageToken=page_token
                ).execute()
                
                results.extend(response.get('files', []))
                page_token = response.get('nextPageToken')
                if not page_token:
                    break
                    
            self.logger.info(f"Found {len(results)} PDF files in folder")
            return results
        except Exception as e:
            self.logger.error(f"Failed to list PDF files: {str(e)}")
            return []

    @lru_cache(maxsize=100)
    def download_pdf(self, file_id: str) -> Optional[bytes]:
        """Download a PDF file from Google Drive."""
        try:
            request = self.drive_service.files().get_media(fileId=file_id)
            file_content = io.BytesIO()
            downloader = io.BytesIO()
            
            response = request.execute()
            downloader.write(response)
            file_content.write(downloader.getvalue())
            file_content.seek(0)
            
            content = file_content.getvalue()
            if not content.startswith(b'%PDF'):
                raise ValueError("Invalid PDF format")
                
            self.logger.debug(f"Successfully downloaded file {file_id}")
            return content
        except Exception as e:
            self.logger.error(f"Failed to download file {file_id}: {str(e)}")
            return None

    def wait_for_server(self, timeout: int = 30, check_interval: float = 0.5) -> bool:
        """Wait for the server to start and become responsive."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(
                    f"http://{self.host}:{self.port}/health",
                    timeout=check_interval
                )
                if response.status_code == 200:
                    return True
            except requests.RequestException:
                time.sleep(check_interval)
        return False

    def stop(self):
        """Stop the vector store server and clean up resources."""
        try:
            if self.server:
                self.server.stop()
            if self.executor:
                self.executor.shutdown(wait=True)
            self.logger.info("Server and resources stopped successfully")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")

def main():
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    pdf_store = None
    try:
        pdf_store = EnhancedPDFVectorStore(
            credentials_file="credentials.json",
            folder_id="1Y2Y0EsMalo26KcJiPYcAXh6UzgMNjh4u",
            debug=True
        )
        
        pdf_store.start()
        
        client = pdf_store.get_client()
        
        # Example query
        results = client.query(
            query="your search query",
            k=5,
            metadata_filter="section_length > 100"
        )
        
        for i, result in enumerate(results, 1):
            logger.info(f"\nResult {i}:")
            logger.info(f"Section: {result['metadata']['heading']}")
            logger.info(f"File: {result['metadata']['file_name']}")
            logger.info(f"Relevance Score: {1 - result['dist']:.3f}")
            logger.info(f"Keywords: {', '.join(result['metadata']['keywords'][:5])}")
            logger.info(f"Preview: {result['text'][:200]}...")
        
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        if pdf_store:
            pdf_store.stop()

if __name__ == "__main__":
    main()