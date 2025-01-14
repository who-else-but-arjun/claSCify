import pandas as pd
import streamlit as st
import torch
import os
from pathlib import Path
import json
import tempfile
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import numpy as np
from PDFparserFITZ import DoraemonPDFParser
from Scibert_embeddings import DoraemonProcessor
from Binary_classification import DoraemonBinaryClassifier
from Conference_classification import DoraemonConferenceClassifier
from Mistral7b_Instruct_1 import Doraemon_justification

def load_css():
    """Load enhanced custom CSS styles"""
    st.markdown("""
        <style>
        /* Modern layout styles */
        .main {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        /* Enhanced card styles */
        .card {
            background-color: white;
            border-radius: 1rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            transition: transform 0.2s;
        }
        
        .card:hover {
            transform: translateY(-2px);
        }
        
        /* Status boxes */
        .success-box {
            background-color: #dcfce7;
            border: 1px solid #86efac;
            border-radius: 1rem;
            padding: 1.5rem;
            margin: 1rem 0;
            animation: fadeIn 0.5s;
        }
        
        .warning-box {
            background-color: #fff7ed;
            border: 1px solid #fed7aa;
            border-radius: 1rem;
            padding: 1.5rem;
            margin: 1rem 0;
            animation: fadeIn 0.5s;
        }
        
        /* Keyword tag style */
        .keyword-tag {
            display: inline-block;
            background-color: #f0f9ff;
            color: #0369a1;
            padding: 8px 16px;
            border-radius: 20px;
            margin: 4px;
            font-size: 0.9em;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        </style>
    """, unsafe_allow_html=True)

def create_gauge_chart(value, title, color_scheme='blues'):
    """Create an enhanced gauge chart with custom color schemes"""
    color_schemes = {
        'blues': ["#cce5ff", "#3b82f6", "#1e40af"],
        'greens': ["#dcfce7", "#22c55e", "#15803d"],
        'oranges': ["#ffedd5", "#f97316", "#9a3412"]
    }
    colors = color_schemes.get(color_scheme, color_schemes['blues'])
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={'text': title, 'font': {'size': 24, 'family': 'Arial, sans-serif'}},
        number={'suffix': "%", 'font': {'size': 28, 'family': 'Arial, sans-serif'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': colors[1]},
            'bgcolor': "white",
            'steps': [
                {'range': [0, 50], 'color': colors[0]},
                {'range': [50, 70], 'color': colors[1]},
                {'range': [70, 100], 'color': colors[2]}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


class ResearchPaperAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pdf_parser = DoraemonPDFParser()
        self.embeddings_processor = DoraemonProcessor()
        
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []
        
        with st.status("🚀 Initializing system...", expanded=True) as status:
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            steps = [
                "Loading PDF processor",
                "Loading embeddings processor",
                "Loading binary classifier",
                "Loading conference classifier",
                "Preparing visualization components"
            ]
            
            for i, step in enumerate(steps):
                progress_text.text(f"⌛ {step}...")
                progress_bar.progress((i + 1) * 20)
                time.sleep(0.5)
            
            self.setup_models()
            progress_bar.progress(100)
            status.update(label="✅ System ready!", state="complete")

    def setup_models(self):
        """Initialize and load models"""
        input_dim = 6176
        self.binary_classifier = DoraemonBinaryClassifier(input_dim=input_dim).to(self.device)
        self.conference_classifier = DoraemonConferenceClassifier(input_dim=input_dim, num_classes=5).to(self.device)
        
        self.load_model(self.binary_classifier, "doraemon_binary_classifier.pt")
        self.load_model(self.conference_classifier, "doraemon_conference_classifier.pt")
        
        self.binary_classifier.eval()
        self.conference_classifier.eval()
        
        self.conference_map = {
            0: "CVPR", 1: "TMLR", 2: "KDD", 
            3: "NEURIPS", 4: "EMNLP"
        }

    def load_model(self, model, checkpoint_path):
        """Load model weights with error handling"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            st.error(f"Error loading model from {checkpoint_path}: {str(e)}")
            raise

    def process_pdf(self, pdf_file):
        """Process PDF and generate analysis"""
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        try:
            def update_progress(percentage, text):
                progress_bar.progress(percentage)
                progress_text.text(text)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save and parse PDF
                temp_path = Path(temp_dir) / "input.pdf"
                update_progress(10, "📥 Saving uploaded file...")
                with open(temp_path, "wb") as f:
                    f.write(pdf_file.getvalue())
                
                update_progress(20, "📄 Parsing PDF content...")
                parsed_content = self.pdf_parser.parse_pdf(str(temp_path))
                
                # Extract text sections
                text = "\n\n".join(f"{heading}\n{content}" 
                                 for heading, content in parsed_content.items())
                abstract = next((content for heading, content in parsed_content.items() 
                               if 'abstract' in heading.lower()), "")
                conclusion = next((content for heading, content in parsed_content.items() 
                                if 'conclusion' in heading.lower()), "")
                
                # Generate embeddings
                update_progress(40, "🔍 Generating embeddings...")
                combined_features, _, _, keywords = self.embeddings_processor.process_document(text)
                
                # Run models
                update_progress(60, "🤖 Running classification models...")
                
                with torch.no_grad():
                    binary_logits = self.binary_classifier(combined_features.unsqueeze(0))
                    binary_prob = torch.sigmoid(binary_logits).item()
                    is_publishable = binary_prob > 0.5
                    
                    conference_pred = None
                    conference_name = None
                    justification = None
                    conference_prob = None
                    
                    if is_publishable:
                        update_progress(80, "🎯 Determining target conference...")
                        conference_logits = self.conference_classifier(combined_features.unsqueeze(0))
                        conference_probs = torch.softmax(conference_logits, dim=1)
                        conference_id = torch.argmax(conference_probs).item()
                        conference_prob = conference_probs[0][conference_id].item()
                        
                        if conference_prob > 0.4:
                            conference_name = self.conference_map[conference_id]
                            
                            update_progress(90, "📝 Generating justification...")
                            justification = Doraemon_justification(
                                abstract=abstract,
                                conclusion=conclusion,
                                keywords=[k[0] for k in keywords],
                                conference_name=conference_name
                            )
                
                update_progress(100, "✅ Analysis complete!")
                
                # Calculate additional metrics
                total_figures = sum(1 for content in parsed_content.values() 
                                  if 'figure' in content.lower())
                total_tables = sum(1 for content in parsed_content.values() 
                                 if 'table' in content.lower())
                
                # Update history
                st.session_state.analysis_history.append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'filename': pdf_file.name,
                    'publishable': "✅ Yes" if is_publishable else "⚠️ No",
                    'conference': conference_name or "N/A",
                    'confidence': f"{binary_prob:.2%}"
                })
                
                return {
                    'is_publishable': is_publishable,
                    'publishable_prob': binary_prob,
                    'conference': conference_name,
                    'conference_prob': conference_prob,
                    'justification': justification,
                    'keywords': [k[0] for k in keywords],
                    'metrics': {
                        'word_count': len(text.split()),
                        'section_count': len(parsed_content),
                        'keyword_count': len(keywords),
                        'total_figures': total_figures,
                        'total_tables': total_tables
                    }
                }
        
        finally:
            progress_bar.empty()
            progress_text.empty()
def main():
    st.set_page_config(
        page_title="claSCIfy : Advanced Research Paper Analyzer",
        layout="wide",
        page_icon="📚"
    )
    load_css()
    
    with st.sidebar:
        st.title("📊 Analytics Dashboard")
        st.markdown("---")
        
        st.subheader("📈 Analysis History")
        if st.session_state.get('analysis_history'):
            history_df = pd.DataFrame(st.session_state.analysis_history)
            st.dataframe(
                history_df,
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No papers analyzed yet")
        
        if st.button("🗑️ Clear History"):
            st.session_state.analysis_history = []
            st.experimental_rerun()
    
    st.title("📚 claSCIfy : Advanced Research Paper Assessment")
    st.markdown("""
    ### 🎯 Analyze your research paper for publication potential and conference fit
    Upload your research paper to get detailed insights about its publication readiness,
    recommended venues, and comprehensive analysis.
    """)
    
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = ResearchPaperAnalyzer()
    
    uploaded_file = st.file_uploader(
        "📤 Upload your research paper (PDF)",
        type="pdf",
        help="Upload a PDF file to analyze its publication potential"
    )
    
    if uploaded_file:
        try:
            results = st.session_state.analyzer.process_pdf(uploaded_file)
            
            st.markdown("## 📊 Analysis Dashboard")
            
            # Top metrics row
            metrics_cols = st.columns(4)
            with metrics_cols[0]:
                st.plotly_chart(
                    create_gauge_chart(results['publishable_prob'], "Publication Potential",
                                     'blues' if results['is_publishable'] else 'oranges'),
                    use_container_width=True
                )
            with metrics_cols[1]:
                st.metric("📝 Word Count", f"{results['metrics']['word_count']:,}",
                         delta="Typical range: 4000-8000")
            with metrics_cols[2]:
                st.metric("📊 Figures & Tables",
                         f"{results['metrics']['total_figures'] + results['metrics']['total_tables']}",
                         delta=f"Figures: {results['metrics']['total_figures']}, Tables: {results['metrics']['total_tables']}")
            with metrics_cols[3]:
                st.metric("📑 Sections", results['metrics']['section_count'],
                         delta=f"Keywords: {results['metrics']['keyword_count']}")
            
            # Main content area
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("## 🎯 Publication Analysis")
                if results['is_publishable']:
                    st.markdown(
                        f"""
                        <div class="success-box">
                            <h3>✅ High Publication Potential</h3>
                            <p>This paper demonstrates strong publication readiness with 
                            {results['publishable_prob']:.1%} confidence.</p>
                            <p>Key strengths:</p>
                            <ul>
                                <li>Well-structured content with {results['metrics']['section_count']} sections</li>
                                <li>Comprehensive analysis supported by {results['metrics']['total_figures']} figures 
                                and {results['metrics']['total_tables']} tables</li>
                                <li>Clear research focus with {results['metrics']['keyword_count']} identified keywords</li>
                            </ul>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    if results['conference']:
                        st.markdown(
                            f"""
                            <div class="card">
                                <h3>🎯 Recommended Venue: {results['conference']}</h3>
                                <p>Confidence: {results['conference_prob']:.1%}</p>
                                <hr>
                                <h4>📝 Submission Rationale:</h4>
                                {results['justification']}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.markdown(
                        f"""
                        <div class="warning-box">
                            <h3>⚠️ Revision Recommended</h3>
                            <p>This paper may benefit from additional refinement before submission. 
                            Current assessment confidence: {results['publishable_prob']:.1%}</p>
                            <p>Consider reviewing:</p>
                            <ul>
                                <li>Content structure and organization</li>
                                <li>Supporting evidence (figures and tables)</li>
                                <li>Research methodology and results presentation</li>
                            </ul>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            with col2:
                st.markdown("### 🏷️ Key Topics")
                keywords_html = " ".join(f"<span class='keyword-tag'>{keyword}</span>" for keyword in results['keywords'])
                st.markdown(keywords_html, unsafe_allow_html=True)
            # Export options
            st.markdown("## 📥 Export Analysis")
            export_col1, export_col2 = st.columns(2)
            with export_col1:
                if st.button("📊 Export Analysis Report"):
                    report = {
                        'filename': uploaded_file.name,
                        'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'publication_potential': {
                            'is_publishable': results['is_publishable'],
                            'confidence': results['publishable_prob'],
                            'recommended_venue': results['conference'],
                            'justification': results['justification']
                        },
                        'metrics': results['metrics'],
                        'keywords': results['keywords']
                    }
                    st.download_button(
                        "📥 Download Report",
                        data=json.dumps(report, indent=2),
                        file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            st.error("Please ensure the PDF is properly formatted and try again.")

if __name__ == "__main__":
    main()
