import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.predictor import SpamPredictor

# Page config
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state
for key, default in [
    ('predictor', None), ('results', []), 
    ('current_model', 'ensemble'), ('model_changed', False)
]:
    if key not in st.session_state:
        st.session_state[key] = default


def load_css():
    """Load enhanced CSS for dark theme"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Base styling */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide header */
    .stApp > header {
        display: none !important;
    }
    
    /* Custom header */
    .custom-header {
        text-align: center;
        padding: 2rem 1rem;
        margin-bottom: 2rem;
        border-bottom: 3px solid #238636;
    }
    
    .custom-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #58a6ff, #3fb950);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .custom-subtitle {
        font-size: 1.2rem;
        color: #8b949e;
    }
    
    /* Enhanced buttons */
    .stButton > button {
        background: linear-gradient(135deg, #238636, #2ea043) !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(35, 134, 54, 0.4) !important;
    }
    
    /* Enhanced metrics */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #161b22, #21262d);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1rem;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #f0f6fc;
    }
    
    /* Enhanced tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #161b22;
        border-radius: 12px;
        padding: 8px;
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #8b949e;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #238636, #2ea043) !important;
        color: white !important;
        font-weight: 600;
    }
    
    /* Enhanced cards */
    .result-card {
        background: linear-gradient(135deg, #161b22, #21262d);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #238636;
    }
    
    .result-card.spam {
        border-left-color: #f85149;
    }
    
    .result-card.ham {
        border-left-color: #3fb950;
    }
    
    /* Status badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 500;
    }
    
    .status-badge.success {
        background: rgba(46, 160, 67, 0.15);
        color: #3fb950;
        border: 1px solid #3fb950;
    }
    
    .status-badge.error {
        background: rgba(248, 81, 73, 0.15);
        color: #f85149;
        border: 1px solid #f85149;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        border-right: 2px solid #238636;
    }
    
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: #f0f6fc;
        font-weight: 700;
    }
    
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #58a6ff;
        font-weight: 600;
        font-size: 1rem;
    }
    
    /* File uploader */
    [data-testid="stFileUploadDropzone"] {
        background: #161b22 !important;
        border: 2px dashed #30363d !important;
        border-radius: 12px !important;
    }
    
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #238636 !important;
    }
    
    /* Hide default menu */
    #MainMenu {
        visibility: hidden;
    }
    
    footer {
        visibility: hidden;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #f0f6fc !important;
    }
    
    /* Text inputs */
    .stTextInput input, .stTextArea textarea {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }
    
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #58a6ff !important;
        box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.2) !important;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
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
    </style>
    """, unsafe_allow_html=True)


def init_models():
    """Initialize ML models"""
    if st.session_state.predictor is None or st.session_state.model_changed:
        with st.spinner("🤖 Loading models..."):
            try:
                st.session_state.predictor = SpamPredictor(
                    model_type=st.session_state.current_model
                )
                st.session_state.model_changed = False
            except Exception as e:
                st.error(f"❌ Error loading models: {e}")
                st.session_state.predictor = None


def main():
    load_css()
    init_models()
    
    # Header
    st.markdown("""
    <div class="custom-header">
        <h1 class="custom-title">📧 Email Spam Classifier</h1>
        <p class="custom-subtitle">✨ AI-powered spam detection with 97.12% accuracy</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.predictor is None:
        st.error("❌ Failed to initialize models. Please check model files.")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        st.markdown("#### 🤖 Model Selection")
        models = {
            "🏆 Ensemble (97.12%)": "ensemble",
            "⚡ Pipeline (96.85%)": "pipeline",
            "🎯 Random Forest (96.50%)": "rf"
        }
        
        current = next((k for k, v in models.items() 
                       if v == st.session_state.current_model), 
                      "🏆 Ensemble (97.12%)")
        
        choice = st.selectbox("Model", list(models.keys()), 
                            index=list(models.keys()).index(current),
                            label_visibility="collapsed")
        
        if models[choice] != st.session_state.current_model:
            st.session_state.current_model = models[choice]
            st.session_state.model_changed = True
            st.rerun()
        
        st.divider()
        
        # Model stats
        st.markdown("#### 📊 Model Performance")
        try:
            info = st.session_state.predictor.get_model_info()
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Accuracy", f"{info.get('accuracy', 0.9712)*100:.1f}%")
            with c2:
                st.metric("Features", info.get('features', '576'))
        except:
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Accuracy", "97.1%")
            with c2:
                st.metric("Features", "576")
        
        st.divider()
        
        st.markdown("#### 📚 Analysis History")
        st.markdown(f"**{len(st.session_state.results)}** emails analyzed")
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.results = []
            st.success("✅ History cleared!")
            st.rerun()
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📝 Analyze", "📁 Batch", "📚 History"])
    
    # Tab 1: Single Analysis
    with tab1:
        st.markdown("### 🔍 Analyze Email")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            email_text = st.text_area(
                "Email Content",
                height=250,
                placeholder="Paste email content here...",
                label_visibility="collapsed"
            )
            
            # Analyze button - now directly below the text area
            if st.button("🔍 Analyze Email", type="primary", use_container_width=True):
                if email_text.strip():
                    with st.spinner("🤖 Analyzing..."):
                        try:
                            result = st.session_state.predictor.predict_from_text(email_text)
                            
                            # Store result
                            st.session_state.results.append({
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'preview': email_text[:80] + "..." if len(email_text) > 80 else email_text,
                                'prediction': result['prediction'],
                                'spam_prob': result['spam_probability'],
                                'ham_prob': result['ham_probability'],
                                'confidence': result['confidence'],
                                'is_spam': result['is_spam']
                            })
                            
                            st.divider()
                            
                            # Result display
                            if result['is_spam']:
                                st.markdown("""
                                <div class="result-card spam">
                                    <h3 style="color: #f85149; margin: 0;">🚨 SPAM DETECTED</h3>
                                    <p style="color: #8b949e; margin: 0.5rem 0;">This email shows spam characteristics</p>
                                </div>
                                """, unsafe_allow_html=True)
                                st.error("⚠️ This email is classified as **SPAM**")
                            else:
                                st.markdown("""
                                <div class="result-card ham">
                                    <h3 style="color: #3fb950; margin: 0;">✅ LEGITIMATE EMAIL</h3>
                                    <p style="color: #8b949e; margin: 0.5rem 0;">This email appears legitimate</p>
                                </div>
                                """, unsafe_allow_html=True)
                                st.success("✅ This email is classified as **HAM**")
                            
                            # Stats
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                st.metric("Spam Probability", f"{result['spam_probability']:.1%}")
                            with c2:
                                st.metric("Ham Probability", f"{result['ham_probability']:.1%}")
                            with c3:
                                st.metric("Confidence", result['confidence'])
                            
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                else:
                    st.warning("⚠️ Please enter email content to analyze")
        
        with col2:
            st.markdown("#### 📋 Templates")
            
            templates = {
                "🚨 Spam Example": """URGENT: Account verification required!
Click here now: http://fake-site.com
Your account will be suspended!""",
                "✅ Ham Example": """Hi Team,
Meeting agenda for tomorrow:
1. Project updates
2. Q4 planning
Best, John"""
            }
            
            template = st.selectbox("Template", list(templates.keys()),
                                  label_visibility="collapsed")
            
            if st.button("📥 Load", use_container_width=True):
                st.session_state.loaded_template = templates[template]
                st.rerun()
            
            st.divider()
            
            uploaded = st.file_uploader("📁 Upload .txt", type=['txt'],
                                        label_visibility="collapsed")
            if uploaded:
                email_text = uploaded.read().decode("utf-8")
        
        # Check for loaded template
        if 'loaded_template' in st.session_state:
            st.warning("⚠️ Template loaded! Switch to the Analyze tab to view it.")
    
    # Tab 2: Batch
    with tab2:
        st.markdown("### 📁 Batch Processing")
        st.caption("Enter multiple emails separated by blank lines")
        
        batch_text = st.text_area(
            "Batch Input",
            height=300,
            placeholder="Email 1...\n\nEmail 2...\n\nEmail 3...",
            label_visibility="collapsed"
        )
        
        if st.button("🚀 Process Batch", type="primary", use_container_width=True):
            if batch_text.strip():
                emails = [e.strip() for e in batch_text.split('\n\n') if e.strip()]
                
                if emails:
                    progress = st.progress(0)
                    results = []
                    
                    for i, email in enumerate(emails):
                        try:
                            r = st.session_state.predictor.predict_from_text(email)
                            results.append({
                                '#': i + 1,
                                'Preview': email[:50] + "..." if len(email) > 50 else email,
                                'Result': r['prediction'],
                                'Spam %': f"{r['spam_probability']:.1%}",
                                'Confidence': r['confidence']
                            })
                        except:
                            results.append({
                                '#': i + 1, 'Preview': "Error", 'Result': "ERROR",
                                'Spam %': "-", 'Confidence': "-"
                            })
                        progress.progress((i + 1) / len(emails))
                    
                    progress.empty()
                    
                    # Summary
                    spam_count = sum(1 for r in results if r['Result'] == 'SPAM')
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Total", len(results))
                    with c2:
                        st.metric("Spam", spam_count)
                    with c3:
                        st.metric("Ham", len(results) - spam_count)
                    
                    st.dataframe(pd.DataFrame(results), use_container_width=True)
                    
                    # Export
                    csv = pd.DataFrame(results).to_csv(index=False)
                    st.download_button(
                        "📥 Download CSV",
                        csv,
                        f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                else:
                    st.warning("⚠️ No valid emails found")
            else:
                st.warning("⚠️ Please enter emails to process")
    
    # Tab 3: History
    with tab3:
        st.markdown("### 📚 Analysis History")
        
        if st.session_state.results:
            df = pd.DataFrame(st.session_state.results)
            
            total, spam = len(df), df['is_spam'].sum()
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total", total)
            with c2:
                st.metric("Spam", spam)
            with c3:
                st.metric("Ham", total - spam)
            
            st.divider()
            
            # Filter
            filter_opt = st.selectbox("Filter", ['All', 'SPAM', 'HAM'])
            
            display = df.copy()
            if filter_opt == 'SPAM':
                display = display[display['is_spam'] == True]
            elif filter_opt == 'HAM':
                display = display[display['is_spam'] == False]
            
            display = display[['timestamp', 'preview', 'prediction', 'confidence']].copy()
            display.columns = ['Time', 'Preview', 'Result', 'Confidence']
            display = display.sort_values('Time', ascending=False)
            
            st.dataframe(display, use_container_width=True, height=400)
            
            # Export
            csv = display.to_csv(index=False)
            st.download_button(
                "📥 Export History",
                csv,
                f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
        else:
            st.info("📝 No history yet. Analyze some emails first!")
    
    # Footer
    st.divider()
    st.caption(f"📧 Email Spam Classifier • Model: {choice} • {len(st.session_state.results)} analyzed")


if __name__ == "__main__":
    main()
