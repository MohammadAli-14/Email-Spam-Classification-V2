"""
app.py - Streamlit web application for email spam classification
Clean, minimal dark theme UI
"""

import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime
import json

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.predictor import SpamPredictor
from src.preprocessing import TextPreprocessor

# Page configuration
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'results' not in st.session_state:
    st.session_state.results = []
if 'current_model' not in st.session_state:
    st.session_state.current_model = "ensemble"
if 'model_changed' not in st.session_state:
    st.session_state.model_changed = False


def apply_dark_theme():
    """Apply clean dark theme CSS"""
    st.markdown("""
    <style>
        /* Base dark theme */
        .stApp {
            background-color: #0d1117;
            color: #c9d1d9;
        }
        
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #161b22 !important;
            border-right: 1px solid #30363d;
        }
        
        section[data-testid="stSidebar"] * {
            color: #c9d1d9 !important;
        }
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #f0f6fc !important;
        }
        
        /* Text */
        p, span, label, div {
            color: #c9d1d9;
        }
        
        /* Input fields */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {
            background-color: #21262d !important;
            color: #c9d1d9 !important;
            border: 1px solid #30363d !important;
            border-radius: 6px !important;
        }
        
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus {
            border-color: #58a6ff !important;
            box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.15) !important;
        }
        
        /* Select boxes */
        .stSelectbox > div > div {
            background-color: #21262d !important;
            border: 1px solid #30363d !important;
            border-radius: 6px !important;
        }
        
        .stSelectbox > div > div > div {
            color: #c9d1d9 !important;
        }
        
        /* Dropdown menus */
        div[data-baseweb="popover"] {
            background-color: #21262d !important;
            border: 1px solid #30363d !important;
        }
        
        div[data-baseweb="popover"] * {
            background-color: #21262d !important;
            color: #c9d1d9 !important;
        }
        
        li[data-baseweb="option"]:hover {
            background-color: #30363d !important;
        }
        
        /* Buttons */
        .stButton > button {
            background-color: #238636 !important;
            color: #ffffff !important;
            border: 1px solid #238636 !important;
            border-radius: 6px !important;
            font-weight: 600 !important;
            transition: background-color 0.2s !important;
        }
        
        .stButton > button:hover {
            background-color: #2ea043 !important;
            border-color: #2ea043 !important;
        }
        
        .stButton > button[kind="secondary"] {
            background-color: #21262d !important;
            border-color: #30363d !important;
            color: #c9d1d9 !important;
        }
        
        .stButton > button[kind="secondary"]:hover {
            background-color: #30363d !important;
        }
        
        /* Metrics */
        div[data-testid="stMetric"] {
            background-color: #161b22 !important;
            border: 1px solid #30363d !important;
            border-radius: 6px !important;
            padding: 1rem !important;
        }
        
        div[data-testid="stMetricValue"] {
            color: #f0f6fc !important;
        }
        
        div[data-testid="stMetricLabel"] {
            color: #8b949e !important;
        }
        
        /* File uploader */
        div[data-testid="stFileUploadDropzone"] {
            background-color: #21262d !important;
            border: 1px dashed #30363d !important;
            border-radius: 6px !important;
        }
        
        div[data-testid="stFileUploadDropzone"] * {
            color: #8b949e !important;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #161b22;
            border-radius: 6px;
            padding: 4px;
            gap: 4px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            color: #8b949e;
            border-radius: 4px;
            padding: 8px 16px;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #21262d;
            color: #c9d1d9;
        }
        
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #21262d;
            color: #f0f6fc;
        }
        
        /* Dataframes */
        .dataframe {
            background-color: #161b22 !important;
        }
        
        .dataframe th {
            background-color: #21262d !important;
            color: #f0f6fc !important;
        }
        
        .dataframe td {
            color: #c9d1d9 !important;
            border-color: #30363d !important;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background-color: #21262d !important;
            border: 1px solid #30363d !important;
            border-radius: 6px !important;
            color: #c9d1d9 !important;
        }
        
        .streamlit-expanderContent {
            background-color: #161b22 !important;
            border: 1px solid #30363d !important;
            border-top: none !important;
        }
        
        /* Alerts */
        .stSuccess {
            background-color: rgba(35, 134, 54, 0.15) !important;
            border: 1px solid #238636 !important;
            color: #3fb950 !important;
        }
        
        .stError {
            background-color: rgba(248, 81, 73, 0.15) !important;
            border: 1px solid #f85149 !important;
            color: #f85149 !important;
        }
        
        .stWarning {
            background-color: rgba(210, 153, 34, 0.15) !important;
            border: 1px solid #d29922 !important;
            color: #d29922 !important;
        }
        
        .stInfo {
            background-color: rgba(88, 166, 255, 0.15) !important;
            border: 1px solid #58a6ff !important;
            color: #58a6ff !important;
        }
        
        /* Dividers */
        hr {
            border-color: #30363d;
        }
        
        /* Result cards */
        .result-card {
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        .spam-result {
            border-left: 4px solid #f85149;
        }
        
        .ham-result {
            border-left: 4px solid #3fb950;
        }
        
        /* Stats grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
        
        .stat-box {
            background-color: #21262d;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 1rem;
            text-align: center;
        }
        
        .stat-label {
            color: #8b949e;
            font-size: 0.85rem;
            margin-bottom: 0.25rem;
        }
        
        .stat-value {
            color: #f0f6fc;
            font-size: 1.5rem;
            font-weight: 600;
        }
        
        /* Hide default streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


def initialize_models():
    """Initialize models once and cache them"""
    if st.session_state.predictor is None or st.session_state.model_changed:
        with st.spinner("Loading model..."):
            try:
                st.session_state.predictor = SpamPredictor(model_type=st.session_state.current_model)
                st.session_state.preprocessor = TextPreprocessor()
                st.session_state.model_changed = False
            except Exception as e:
                st.error(f"Error loading models: {e}")
                st.session_state.predictor = None


def main():
    # Apply theme
    apply_dark_theme()
    
    # Initialize models
    initialize_models()
    
    # Header
    st.title("📧 Email Spam Classifier")
    st.caption("AI-powered spam detection with 97.12% accuracy")
    
    if st.session_state.predictor is None:
        st.error("Failed to initialize models. Please check if model files exist.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        st.subheader("Model")
        
        model_map = {
            "Ensemble (97.12%)": "ensemble",
            "Pipeline (96.85%)": "pipeline", 
            "Random Forest (96.50%)": "rf"
        }
        
        current_display = next(
            (name for name, key in model_map.items() if key == st.session_state.current_model),
            "Ensemble (97.12%)"
        )
        
        model_choice = st.selectbox(
            "Select model:",
            options=list(model_map.keys()),
            index=list(model_map.keys()).index(current_display),
            label_visibility="collapsed"
        )
        
        if model_map[model_choice] != st.session_state.current_model:
            st.session_state.current_model = model_map[model_choice]
            st.session_state.model_changed = True
            st.rerun()
        
        st.divider()
        
        # Model info
        st.subheader("📊 Model Info")
        try:
            model_info = st.session_state.predictor.get_model_info()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", f"{model_info.get('accuracy', 0.9712)*100:.1f}%")
            with col2:
                st.metric("Features", model_info.get('features', '576'))
        except:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", "97.1%")
            with col2:
                st.metric("Features", "576")
        
        st.divider()
        
        # History actions
        st.subheader("📚 History")
        st.write(f"**{len(st.session_state.results)}** emails analyzed")
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.results = []
            st.success("History cleared!")
            st.rerun()
    
    # Main content - Tabs
    tab1, tab2, tab3 = st.tabs(["📝 Analyze", "📁 Batch", "📚 History"])
    
    # Tab 1: Single Email Analysis
    with tab1:
        st.subheader("Analyze Email")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            email_input = st.text_area(
                "Email Content",
                height=250,
                placeholder="Paste the email content here...",
                label_visibility="collapsed"
            )
        
        with col2:
            st.markdown("**Quick Templates**")
            
            templates = {
                "Spam Example": """URGENT: Your account needs verification!
                
Click here immediately to verify your PayPal account or it will be suspended.

http://fake-verify.com

Act now!""",
                "Ham Example": """Hi Team,

Here's the agenda for tomorrow's meeting:
1. Project updates
2. Q4 planning
3. Team assignments

Best regards,
John"""
            }
            
            selected = st.selectbox("Choose:", list(templates.keys()), label_visibility="collapsed")
            
            if st.button("Load Template", use_container_width=True):
                st.session_state.template_text = templates[selected]
                st.rerun()
            
            st.divider()
            
            uploaded = st.file_uploader("Upload .txt", type=['txt'], label_visibility="collapsed")
            if uploaded:
                email_input = uploaded.read().decode("utf-8")
        
        # Check for template
        if 'template_text' in st.session_state:
            email_input = st.session_state.template_text
            del st.session_state.template_text
        
        # Analyze button
        if st.button("🔍 Analyze Email", type="primary", use_container_width=True):
            if email_input and email_input.strip():
                with st.spinner("Analyzing..."):
                    try:
                        result = st.session_state.predictor.predict_from_text(email_input)
                        
                        # Store result
                        st.session_state.results.append({
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'preview': email_input[:80] + "..." if len(email_input) > 80 else email_input,
                            'prediction': result['prediction'],
                            'spam_prob': result['spam_probability'],
                            'ham_prob': result['ham_probability'],
                            'confidence': result['confidence'],
                            'is_spam': result['is_spam']
                        })
                        
                        # Display result
                        st.divider()
                        
                        if result['is_spam']:
                            st.markdown("""
                            <div class="result-card spam-result">
                                <h3 style="color: #f85149; margin: 0;">🚨 SPAM DETECTED</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.error("This email is classified as **SPAM**")
                        else:
                            st.markdown("""
                            <div class="result-card ham-result">
                                <h3 style="color: #3fb950; margin: 0;">✅ LEGITIMATE EMAIL</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.success("This email is classified as **HAM** (legitimate)")
                        
                        # Stats
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Spam Probability", f"{result['spam_probability']:.1%}")
                        with col2:
                            st.metric("Ham Probability", f"{result['ham_probability']:.1%}")
                        with col3:
                            st.metric("Confidence", result['confidence'])
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter email content to analyze.")
    
    # Tab 2: Batch Processing
    with tab2:
        st.subheader("Batch Processing")
        st.caption("Enter multiple emails separated by empty lines")
        
        batch_input = st.text_area(
            "Batch Email Content",
            height=300,
            placeholder="Email 1 content...\n\nEmail 2 content...\n\nEmail 3 content...",
            label_visibility="collapsed"
        )
        
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("🚀 Process Batch", type="primary", use_container_width=True):
                if batch_input and batch_input.strip():
                    emails = [e.strip() for e in batch_input.split('\n\n') if e.strip()]
                    
                    if emails:
                        progress = st.progress(0)
                        results = []
                        
                        for i, email in enumerate(emails):
                            try:
                                result = st.session_state.predictor.predict_from_text(email)
                                results.append({
                                    '#': i + 1,
                                    'Preview': email[:50] + "..." if len(email) > 50 else email,
                                    'Result': result['prediction'],
                                    'Spam %': f"{result['spam_probability']:.1%}",
                                    'Confidence': result['confidence']
                                })
                            except:
                                results.append({
                                    '#': i + 1,
                                    'Preview': "Error",
                                    'Result': "ERROR",
                                    'Spam %': "-",
                                    'Confidence': "-"
                                })
                            progress.progress((i + 1) / len(emails))
                        
                        progress.empty()
                        
                        # Summary
                        spam_count = sum(1 for r in results if r['Result'] == 'SPAM')
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total", len(results))
                        with col2:
                            st.metric("Spam", spam_count)
                        with col3:
                            st.metric("Ham", len(results) - spam_count)
                        
                        # Results table
                        st.dataframe(pd.DataFrame(results), use_container_width=True)
                        
                        # Export
                        csv = pd.DataFrame(results).to_csv(index=False)
                        st.download_button(
                            "📥 Download CSV",
                            csv,
                            f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv",
                            use_container_width=True
                        )
                    else:
                        st.warning("No valid emails found.")
                else:
                    st.warning("Please enter emails to process.")
        
        with col2:
            uploaded_batch = st.file_uploader("Upload file", type=['txt', 'csv'], key="batch_upload")
            if uploaded_batch:
                try:
                    content = uploaded_batch.read().decode("utf-8")
                    st.info(f"Loaded {len(content.split(chr(10)+chr(10)))} emails")
                except:
                    st.error("Error reading file")
    
    # Tab 3: History
    with tab3:
        st.subheader("Analysis History")
        
        if st.session_state.results:
            df = pd.DataFrame(st.session_state.results)
            
            # Summary stats
            total = len(df)
            spam = df['is_spam'].sum()
            ham = total - spam
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Analyzed", total)
            with col2:
                st.metric("Spam", spam)
            with col3:
                st.metric("Ham", ham)
            
            st.divider()
            
            # Filter
            filter_type = st.selectbox("Filter:", ['All', 'SPAM', 'HAM'], label_visibility="collapsed")
            
            display_df = df.copy()
            if filter_type == 'SPAM':
                display_df = display_df[display_df['is_spam'] == True]
            elif filter_type == 'HAM':
                display_df = display_df[display_df['is_spam'] == False]
            
            # Display
            display_df = display_df[['timestamp', 'preview', 'prediction', 'confidence']].copy()
            display_df.columns = ['Time', 'Preview', 'Result', 'Confidence']
            display_df = display_df.sort_values('Time', ascending=False)
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
            # Export
            csv = display_df.to_csv(index=False)
            st.download_button(
                "📥 Export History",
                csv,
                f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
        else:
            st.info("No analysis history yet. Start by analyzing some emails!")
    
    # Footer
    st.divider()
    st.caption(f"📧 Email Spam Classifier • Model: {model_choice} • {len(st.session_state.results)} emails analyzed")


if __name__ == "__main__":
    main()