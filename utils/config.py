"""
Configuration file for Spam Detection Application
Contains theme settings, app configuration, and constants
"""

import streamlit as st

# =============================================================================
# THEME & COLOR CONFIGURATION
# =============================================================================

class Theme:
    """Professional blue theme configuration"""
    
    # Primary Colors
    PRIMARY_BLUE = "#1E3A8A"      # Deep professional blue
    SECONDARY_BLUE = "#3B82F6"    # Bright blue
    LIGHT_BLUE = "#EFF6FF"        # Very light blue background
    ACCENT_BLUE = "#1D4ED8"       # Accent blue
    
    # Supporting Colors
    SUCCESS_GREEN = "#059669"     # Success messages
    WARNING_ORANGE = "#D97706"    # Warning messages
    ERROR_RED = "#DC2626"         # Error messages
    TEXT_DARK = "#1F2937"         # Dark text
    TEXT_LIGHT = "#6B7280"        # Light text
    BACKGROUND_WHITE = "#FFFFFF"  # White background
    BORDER_GRAY = "#E5E7EB"       # Border color

    # Gradients
    BLUE_GRADIENT = "linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%)"
    LIGHT_GRADIENT = "linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%)"

class AppConfig:
    """Application configuration settings"""
    
    # App Info
    APP_NAME = "🛡️ Spam Detection System"
    APP_VERSION = "1.0.0"
    APP_DESCRIPTION = "Professional AI-powered email classification system"
    
    # Page Configuration
    PAGE_TITLE = "Spam Detection | AI Classification"
    PAGE_ICON = "🛡️"
    LAYOUT = "wide"
    INITIAL_SIDEBAR_STATE = "expanded"
    
    # Model Configuration
    MODELS = {
        "SVM": {
            "name": "Support Vector Machine",
            "file": "linear_svm_tfidf_calibrated.pkl",
            "description": "Calibrated SVM with TF-IDF vectorization",
            "color": Theme.PRIMARY_BLUE
        },
        "LR": {
            "name": "Logistic Regression", 
            "file": "logistic_regression_tfidf.pkl",
            "description": "Logistic Regression with TF-IDF vectorization",
            "color": Theme.SECONDARY_BLUE
        }
    }
    
    # UI Settings
    SIDEBAR_WIDTH = 300
    MAIN_CONTAINER_PADDING = "2rem"
    CARD_BORDER_RADIUS = "12px"
    BUTTON_BORDER_RADIUS = "8px"

# =============================================================================
# CUSTOM CSS STYLES
# =============================================================================

def get_custom_css():
    """Returns custom CSS for professional blue theme"""
    return f"""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {{
        font-family: 'Inter', sans-serif;
        background: {Theme.LIGHT_BLUE};
    }}
    
    /* Main Header */
    .main-header {{
        background: {Theme.BLUE_GRADIENT};
        padding: 2rem 0;
        border-radius: {AppConfig.CARD_BORDER_RADIUS};
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(30, 58, 138, 0.15);
    }}
    
    .main-title {{
        color: white;
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    
    .main-subtitle {{
        color: rgba(255,255,255,0.9);
        text-align: center;
        font-size: 1.2rem;
        font-weight: 400;
        margin: 0.5rem 0 0 0;
    }}
    
    /* Cards */
    .metric-card {{
        background: {Theme.BACKGROUND_WHITE};
        padding: 1.5rem;
        border-radius: {AppConfig.CARD_BORDER_RADIUS};
        border: 1px solid {Theme.BORDER_GRAY};
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }}
    
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(30, 58, 138, 0.15);
    }}
    
    .prediction-card {{
        background: {Theme.BACKGROUND_WHITE};
        padding: 2rem;
        border-radius: {AppConfig.CARD_BORDER_RADIUS};
        border-left: 4px solid {Theme.PRIMARY_BLUE};
        box-shadow: 0 2px 15px rgba(0,0,0,0.08);
        margin: 1.5rem 0;
    }}
    
    /* Buttons */
    .stButton > button {{
        background: {Theme.BLUE_GRADIENT};
        color: white;
        border: none;
        border-radius: {AppConfig.BUTTON_BORDER_RADIUS};
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 10px rgba(30, 58, 138, 0.2);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(30, 58, 138, 0.3);
    }}
    
    /* Sidebar */
    .css-1d391kg {{
        background: {Theme.BACKGROUND_WHITE};
        border-right: 2px solid {Theme.BORDER_GRAY};
    }}
    
    /* Success/Error Messages */
    .success-message {{
        background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%);
        color: {Theme.SUCCESS_GREEN};
        padding: 1rem;
        border-radius: {AppConfig.BUTTON_BORDER_RADIUS};
        border-left: 4px solid {Theme.SUCCESS_GREEN};
        margin: 1rem 0;
    }}
    
    .error-message {{
        background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
        color: {Theme.ERROR_RED};
        padding: 1rem;
        border-radius: {AppConfig.BUTTON_BORDER_RADIUS};
        border-left: 4px solid {Theme.ERROR_RED};
        margin: 1rem 0;
    }}
    
    /* Model Comparison Cards */
    .model-card {{
        background: {Theme.BACKGROUND_WHITE};
        padding: 1.5rem;
        border-radius: {AppConfig.CARD_BORDER_RADIUS};
        margin: 1rem 0;
        border: 2px solid {Theme.BORDER_GRAY};
        transition: all 0.3s ease;
    }}
    
    .model-card:hover {{
        border-color: {Theme.PRIMARY_BLUE};
        transform: scale(1.02);
    }}
    
    .model-card.svm {{
        border-left: 4px solid {Theme.PRIMARY_BLUE};
    }}
    
    .model-card.lr {{
        border-left: 4px solid {Theme.SECONDARY_BLUE};
    }}
    
    /* Hide Streamlit Elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    /* Progress Bar */
    .stProgress > div > div > div > div {{
        background: {Theme.BLUE_GRADIENT};
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 24px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: {Theme.LIGHT_BLUE};
        color: {Theme.TEXT_DARK};
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        font-weight: 600;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {Theme.PRIMARY_BLUE};
        color: white;
    }}
    </style>
    """

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def apply_theme():
    """Apply the custom theme to the Streamlit app"""
    st.markdown(get_custom_css(), unsafe_allow_html=True)

def create_header(title, subtitle=None):
    """Create a professional header with gradient background"""
    header_html = f"""
    <div class="main-header">
        <h1 class="main-title">{title}</h1>
        {f'<p class="main-subtitle">{subtitle}</p>' if subtitle else ''}
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

def create_metric_card(title, value, description="", color=Theme.PRIMARY_BLUE):
    """Create a metric card with professional styling"""
    card_html = f"""
    <div class="metric-card">
        <h3 style="color: {color}; margin: 0 0 0.5rem 0; font-size: 1.1rem; font-weight: 600;">{title}</h3>
        <p style="font-size: 2rem; font-weight: 700; margin: 0; color: {Theme.TEXT_DARK};">{value}</p>
        {f'<p style="color: {Theme.TEXT_LIGHT}; margin: 0.5rem 0 0 0; font-size: 0.9rem;">{description}</p>' if description else ''}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

# =============================================================================
# CONSTANTS
# =============================================================================

# Text preprocessing constants
MAX_MESSAGE_LENGTH = 5000
MIN_MESSAGE_LENGTH = 5

# Batch processing limits
MAX_BATCH_SIZE = 100

# Cache settings (in seconds)
MODEL_CACHE_TTL = 3600  # 1 hour