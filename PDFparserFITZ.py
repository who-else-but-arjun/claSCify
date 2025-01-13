import fitz
import json
import re
from typing import List, Dict, Tuple, Optional
import os
from collections import defaultdict
import logging
from datetime import datetime

class DoraemonPDFParser:
    def __init__(self, min_section_length: int = 50):
        self.patterns = {
            'urls': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'citations': r'\[[0-9,\s-]+\]',
            'references': r'^(?:References?|REFERENCES?|Bibliography|BIBLIOGRAPHY)(?:\s|$)',
            'appendix': r'^(?:Appendix|APPENDIX)(?:\s+[A-Z])?(?:\s|:|$)',
            'acknowledgments': r'^(?:Acknowledgments?|ACKNOWLEDGMENTS?)(?:\s|$)',
            'emails': r'[\w\.-]+@[\w\.-]+\.\w+',
            'line_numbers': r'^\d+$',
            'page_numbers': r'^\d+$',
            'cross_refs': r'(Fig\.|Figure|Table|Section)\s*\d+',
            'figure_captions': r'(Figure|Fig\.)\s*\d+[.:]\s*.*?(?=\n|$)',
            'table_captions': r'Table\s*\d+[.:]\s*.*?(?=\n|$)',
            'section_number': r'^(?:\d+\.)*\d+(?:\s+|\b)|^\.\d+(?:\s+|\b)'
        }
        self.heading_font_sizes = []
        self.found_references = False
        self.found_appendix = False
        self.min_section_length = min_section_length
        self.setup_logging()

    def setup_logging(self):
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #log_file = os.path.join(log_dir, f"pdf_parser_{timestamp}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def clean_text(self, text: str) -> str:
        text = text.encode('ascii', 'ignore').decode('ascii')
        for pattern_name, pattern in self.patterns.items():
            if pattern_name not in ['figure_captions', 'table_captions', 'section_number']:
                text = re.sub(pattern, '', text)
        text = re.sub(r'(?<=[a-z])-\n(?=[a-z])', '', text)
        text = re.sub(r'(?<=[a-z])\n(?=[a-z])', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def get_font_properties(self, block: Dict) -> Tuple[str, float, bool, str]:
        text = ""
        max_font_size = 0
        is_bold = False
        font_face = ""
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text += span.get("text", "")
                current_size = span.get("size", 0)
                if current_size > max_font_size:
                    max_font_size = current_size
                    font_face = span.get("font", "")
                is_bold = is_bold or (span.get("flags", 0) & 2 ** 2 != 0)
        return text, max_font_size, is_bold, font_face

    def clean_heading(self, text: str) -> str:
        text = re.sub(r'^(?:\d+\.)*\d+(?:\s+|\b)|^\.\d+(?:\s+|\b)', '', text)
        text = re.sub(r'^(?:\d+\.)*\d+([A-Z][a-z])', r'\1', text)
        text = re.sub(r'^\.\d+([A-Z][a-z])', r'\1', text)
        text = re.sub(r'\.+\s*$', '', text)
        text = re.sub(r'^\d+([A-Z])', r'\1', text)
        text = re.sub(r'^\.\d+([A-Z])', r'\1', text)
        return text.strip()

    def is_heading(self, text: str, font_size: float, is_bold: bool, font_face: str) -> Tuple[bool, str]:
        if re.match(self.patterns['references'], text):
            self.found_references = True
            return False, text
        elif re.match(self.patterns['appendix'], text):
            self.found_appendix = True
            return False, text
            
        heading_patterns = [
            r'^(?:\d+\.)*\d+\s+[A-Z][A-Za-z\s]+$',
            r'^(?:\d+\.)*\d+\s+[A-Z][A-Z\s]+$',
            r'^(?:\d+\.)*\d+[A-Z][A-Za-z\s]+$',
            r'^\.\d+\s+[A-Z][A-Za-z\s]+$',
            r'^\.\d+[A-Z][A-Za-z\s]+$',
            r'^[A-Z][A-Z\s]{3,}[A-Z]$',
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$'
        ]
        
        if len(self.heading_font_sizes) < 10:
            self.heading_font_sizes.append(font_size)
        
        avg_font_size = sum(self.heading_font_sizes) / len(self.heading_font_sizes) if self.heading_font_sizes else 11
        is_larger_font = font_size > avg_font_size + 1
        text = text.strip()
        
        is_pattern_match = any(re.match(pattern, text) for pattern in heading_patterns)
        is_short = len(text) < 200
        has_heading_properties = (is_larger_font or is_bold) and is_short
        
        if is_pattern_match or has_heading_properties:
            cleaned_heading = self.clean_heading(text)
            return True, cleaned_heading
        return False, text

    def should_include_section(self, heading: str, text: str) -> bool:
        if self.found_references or self.found_appendix:
            return False
            
        if any(re.match(self.patterns[pattern], heading.strip()) 
               for pattern in ['references', 'appendix', 'acknowledgments']):
            if re.match(self.patterns['references'], heading.strip()):
                self.found_references = True
            elif re.match(self.patterns['appendix'], heading.strip()):
                self.found_appendix = True
            return False
            
        if len(text.strip().split()) < self.min_section_length:
            return False
            
        return True

    def parse_pdf(self, pdf_path: str) -> Dict:
        try:
            doc = fitz.open(pdf_path)
            sections = {}
            current_heading = ''
            current_content = []
            self.heading_font_sizes = []
            self.found_references = False
            self.found_appendix = False
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if block.get("type") == 0:
                        text, font_size, is_bold, font_face = self.get_font_properties(block)
                        text = self.clean_text(text)
                        if not text:
                            continue
                        is_heading, heading_text = self.is_heading(text, font_size, is_bold, font_face)
                        if is_heading:
                            if current_heading and current_content:
                                section_text = ' '.join(current_content)
                                if self.should_include_section(current_heading, section_text):
                                    sections[current_heading] = section_text
                            current_heading = heading_text
                            current_content = []
                        else:
                            current_content.append(text)
            
            if current_heading and current_content:
                section_text = ' '.join(current_content)
                if self.should_include_section(current_heading, section_text):
                    sections[current_heading] = section_text
            
            doc.close()
            return sections
            
        except Exception as e:
            self.logger.error(f"Error parsing PDF {pdf_path}: {str(e)}")
            return {}

    def save_to_json(self, parsed_content: Dict, output_path: str):
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(parsed_content, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Successfully saved parsed content to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving to JSON: {str(e)}")

def process_directory(input_dir: str, output_dir: str, min_section_length: int = 50):
    parser = DoraemonPDFParser(min_section_length=min_section_length)
    count = 1
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                output_path = os.path.join(
                    output_dir,
                    relative_path,
                    f"P{count:03d}.json"
                )
                count += 1
                #parser.logger.info(f"Processing: {pdf_path}")
                parsed_content = parser.parse_pdf(pdf_path)
                parser.save_to_json(parsed_content, output_path)

if __name__ == "__main__":
    base_input_dir = "Sample/pdfs/"
    base_output_dir = "Sample/texts/"
    input_dir = os.path.join(base_input_dir)
    output_dir = os.path.join(base_output_dir)
    process_directory(input_dir, output_dir)