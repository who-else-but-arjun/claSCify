import pandas as pd
import streamlit as st
import torch
import os
from pathlib import Path
import json
import tempfile
from PDFparserFITZ import DoraemonPDFParser
from Scibert_embeddings import DoraemonProcessor
from Binary_classification import DoraemonBinaryClassifier
from Conference_classification import DoraemonConferenceClassifier
from Mistral7b_Instruct_2 import Doraemon_justification

class ResearchPaperAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pdf_parser = DoraemonPDFParser()
        self.embeddings_processor = DoraemonProcessor()
        self.setup_models()
    def load_model(self,model, checkpoint_path):
        """Helper function to load model with correct state dict structure"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        return model
    def setup_models(self):
        # Initialize classifiers
        input_dim = 6176 # Combined feature vector size
        # Load model weights
        self.binary_classifier = self.load_model(DoraemonBinaryClassifier(input_dim=input_dim).to(self.device),"doraemon_binary_classifier.pt")
        self.conference_classifier = self.load_model(DoraemonConferenceClassifier(input_dim=input_dim, num_classes=5).to(self.device),"doraemon_conference_classifier.pt")
    
        self.binary_classifier.eval()
        self.conference_classifier.eval()
        
        self.conference_map = {0: "CVPR", 1: "TMLR", 2: "KDD", 3: "NEURIPS", 4: "EMNLP"}

    def process_pdf(self, pdf_file):
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pdf_path = temp_path / "input.pdf"
            
            # Save uploaded file
            with open(pdf_path, "wb") as f:
                f.write(pdf_file.getvalue())
            
            # Parse PDF
            parsed_content = self.pdf_parser.parse_pdf(str(pdf_path))
            
            # Extract text for embeddings
            text = ""
            for heading, content in parsed_content.items():
                text += f"{heading}\n{content}\n\n"
            
            # Generate embeddings and features
            combined_features, _, _, keywords = self.embeddings_processor.process_document(text)
            
            # Get predictions
            with torch.no_grad():
                binary_pred = self.binary_classifier(combined_features.unsqueeze(0))
                is_publishable = binary_pred.item() > 0.5
                
                conference_pred = None
                conference_name = None
                justification = None
                
                if is_publishable:
                    conference_pred = self.conference_classifier(combined_features.unsqueeze(0))
                    conference_id = torch.argmax(conference_pred).item()
                    conference_name = self.conference_map[conference_id]
                    
                    # Get abstract and conclusion from parsed content
                    abstract = ""
                    conclusion = ""
                    for heading, content in parsed_content.items():
                        if 'abstract' in heading.lower():
                            abstract = content
                        elif 'conclusion' in heading.lower():
                            conclusion = content
                    
                    justification = Doraemon_justification(
                        abstract=abstract,
                        conclusion=conclusion,
                        keywords=[k[0] for k in keywords],
                        conference_name=conference_name
                    )
            
            return {
                'is_publishable': is_publishable,
                'conference': conference_name,
                'justification': justification,
                'keywords': keywords,
                'sections': parsed_content
            }

def main():
    st.set_page_config(page_title="Research Paper Analyzer", layout="wide")
    
    # Header
    st.title("📚 Research Paper Analysis Dashboard")
    st.markdown("---")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = ResearchPaperAnalyzer()
    
    # File upload
    uploaded_file = st.file_uploader("Upload your research paper (PDF)", type="pdf")
    
    if uploaded_file:
        with st.spinner("Analyzing paper... This may take a few minutes."):
            try:
                results = st.session_state.analyzer.process_pdf(uploaded_file)
                
                # Display results in columns
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.header("📊 Analysis Results")
                    
                    # Publishability prediction
                    if results['is_publishable']:
                        st.success("✅ This paper is likely publishable!")
                        st.subheader(f"Recommended Venue: {results['conference']}")
                        
                        # Display justification
                        st.markdown("### 📝 Justification")
                        st.write(results['justification'])
                    else:
                        st.error("⚠️ This paper may need revision before submission")
                
                with col2:
                    # Keywords section
                    st.header("🔑 Key Topics")
                    keywords_df = pd.DataFrame(results['keywords'], columns=['Keyword', 'Relevance'])
                    keywords_df['Relevance'] = keywords_df['Relevance'].apply(lambda x: f"{x:.2%}")
                    st.dataframe(keywords_df)
                
                # Expandable section for paper structure
                with st.expander("📄 Paper Structure"):
                    for heading, content in results['sections'].items():
                        st.markdown(f"### {heading}")
                        st.write(content[:500] + "..." if len(content) > 500 else content)
                
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                st.error("Please make sure the PDF is properly formatted and try again.")

if __name__ == "__main__":
    main()