"""
Spam Detection System - Main Application
Professional AI-powered email classification system with multiple models
"""

import streamlit as st
import sys
import os
from datetime import datetime

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from utils.config import AppConfig, Theme, apply_theme, create_header
from models.predictor import get_predictor

# Import UI pages
from ui.home import show_home_page
from ui.prediction import show_prediction_page
from ui.batch_analysis import show_batch_analysis_page

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title=AppConfig.PAGE_TITLE,
    page_icon=AppConfig.PAGE_ICON,
    layout=AppConfig.LAYOUT,
    initial_sidebar_state=AppConfig.INITIAL_SIDEBAR_STATE,
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': f"{AppConfig.APP_NAME} v{AppConfig.APP_VERSION}"
    }
)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def initialize_session_state():
    """Initialize session state variables"""
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
    
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
    
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home"
    
    if 'theme_loaded' not in st.session_state:
        st.session_state.theme_loaded = False

# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================

def create_sidebar():
    """Create professional sidebar with navigation and system info"""
    
    with st.sidebar:
        # App Logo and Title
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem 0; background: {Theme.BLUE_GRADIENT}; 
                    border-radius: 12px; margin-bottom: 2rem;">
            <h2 style="color: white; margin: 0; font-size: 1.5rem; font-weight: 700;">
                🛡️ Spam Detector
            </h2>
            <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                v{AppConfig.APP_VERSION}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation Menu
        st.markdown("### 🧭 Navigation")
        
        # Page selection with custom styling
        pages = {
            "🏠 Home": "Home",
            "🔍 Single Prediction": "Prediction", 
            "📊 Batch Analysis": "Batch Analysis"
        }
        
        # Create navigation buttons
        for display_name, page_key in pages.items():
            if st.button(
                display_name,
                use_container_width=True,
                key=f"nav_{page_key}",
                type="primary" if st.session_state.current_page == page_key else "secondary"
            ):
                st.session_state.current_page = page_key
                st.rerun()
        
        st.divider()
        
        # System Status
        st.markdown("### 🔧 System Status")
        
        # Model Status
        if st.session_state.models_loaded and st.session_state.predictor:
            try:
                stats = st.session_state.predictor.get_model_stats()
                
                # Model availability
                st.markdown(f"""
                <div style="background: {Theme.LIGHT_BLUE}; padding: 1rem; border-radius: 8px; 
                            border-left: 4px solid {Theme.SUCCESS_GREEN};">
                    <strong style="color: {Theme.SUCCESS_GREEN};">✅ Models Loaded</strong><br>
                    <small style="color: {Theme.TEXT_LIGHT};">
                        {stats['total_models']} model(s) available
                    </small>
                </div>
                """, unsafe_allow_html=True)
                
                # Individual model status
                for model_key, model_info in stats['model_info'].items():
                    model_color = Theme.PRIMARY_BLUE if model_key == 'svm' else Theme.SECONDARY_BLUE
                    st.markdown(f"""
                    <div style="margin: 0.5rem 0; padding: 0.5rem; background: white; 
                                border-radius: 6px; border-left: 3px solid {model_color};">
                        <strong style="color: {model_color}; font-size: 0.9rem;">
                            {model_info['name']}
                        </strong><br>
                        <small style="color: {Theme.TEXT_LIGHT};">
                            {model_info.get('type', 'Unknown')}
                        </small>
                    </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error getting model stats: {str(e)}")
        else:
            st.markdown(f"""
            <div style="background: {Theme.LIGHT_BLUE}; padding: 1rem; border-radius: 8px; 
                        border-left: 4px solid {Theme.WARNING_ORANGE};">
                <strong style="color: {Theme.WARNING_ORANGE};">⚠️ Loading Models...</strong><br>
                <small style="color: {Theme.TEXT_LIGHT};">Please wait</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Prediction History Summary
        if st.session_state.prediction_history:
            st.markdown("### 📈 Session Stats")
            
            total_predictions = len(st.session_state.prediction_history)
            spam_count = sum(1 for p in st.session_state.prediction_history 
                           if p.get('prediction_label') == 'SPAM')
            ham_count = total_predictions - spam_count
            
            # Stats display
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total", total_predictions)
            with col2:
                st.metric("Spam", spam_count)
            
            # Clear history button
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.prediction_history = []
                st.success("History cleared!")
                st.rerun()
        
        # Footer
        st.markdown("---")
        st.markdown(f"""
        <div style="text-align: center; color: {Theme.TEXT_LIGHT}; font-size: 0.8rem;">
            <p>🤖 AI-Powered Classification</p>
            <p>Built with Streamlit</p>
            <p>{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# MAIN APPLICATION LOGIC
# =============================================================================

def load_models():
    """Load models with progress indication"""
    if not st.session_state.models_loaded:
        with st.spinner("🚀 Initializing AI models..."):
            try:
                predictor = get_predictor()
                available_models = predictor.get_available_models()
                if available_models:
                    st.session_state.predictor = predictor
                    st.session_state.models_loaded = True
                    st.success(f"✅ Successfully loaded {len(available_models)} model(s)!", icon="🤖")
                else:
                    st.error("❌ No models could be loaded. Please check model files.", icon="🚨")
                    st.stop()
                    
            except Exception as e:
                st.error(f"❌ Failed to load models: {str(e)}", icon="🚨")
                st.info("💡 Make sure model files are in the 'models/' directory")
                st.stop()

def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Apply custom theme
    if not st.session_state.theme_loaded:
        apply_theme()
        st.session_state.theme_loaded = True
    
    # Create sidebar
    create_sidebar()
    
    # Load models if not loaded
    load_models()
    
    # Main content area
    with st.container():
        # Route to appropriate page based on selection
        current_page = st.session_state.current_page
        
        if current_page == "Home":
            show_home_page()
        elif current_page == "Prediction":
            show_prediction_page()
        elif current_page == "Batch Analysis":
            show_batch_analysis_page()
        else:
            st.error(f"Unknown page: {current_page}")
    
    # Add some spacing at bottom
    st.markdown("<br><br>", unsafe_allow_html=True)

# =============================================================================
# ERROR HANDLING
# =============================================================================

def handle_errors():
    """Global error handler"""
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}", icon="🚨")
        st.info("🔄 Please refresh the page and try again")
        
        # Show error details in expander for debugging
        with st.expander("🐛 Error Details (for debugging)"):
            st.code(str(e))
            
        # Emergency reset button
        if st.button("🆘 Reset Application"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("Application reset! Please refresh the page.")

# =============================================================================
# RUN APPLICATION
# =============================================================================

if __name__ == "__main__":
    try:
        handle_errors()
    except Exception as e:
        # Last resort error handling
        st.markdown(f"""
        <div style="background: #FEE2E2; border: 1px solid #DC2626; border-radius: 8px; 
                    padding: 1rem; margin: 1rem 0;">
            <h3 style="color: #DC2626; margin: 0;">🚨 Critical Error</h3>
            <p>The application encountered a critical error and cannot continue.</p>
            <code>{str(e)}</code>
            <p><strong>Please refresh the page and ensure all required files are present.</strong></p>
        </div>
        """, unsafe_allow_html=True)