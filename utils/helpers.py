"""
Utility Functions and Helpers for Spam Detection System
Professional helper functions for UI components, data handling, and system utilities
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Union, Tuple, Any, Optional
import json
import io
import base64
import hashlib
import logging
from pathlib import Path
import time
import re

# =============================================================================
# UI HELPER FUNCTIONS
# =============================================================================

def create_metric_card(title: str, value: Union[str, int, float], 
                      delta: Optional[str] = None, 
                      delta_color: str = "normal",
                      icon: Optional[str] = None,
                      background_color: str = "#1E3A8A") -> None:
    """
    Create a professional metric card with custom styling
    
    Args:
        title: Card title
        value: Main metric value
        delta: Optional delta value
        delta_color: Color for delta (normal, inverse, off)
        icon: Optional emoji icon
        background_color: Background color for the card
    """
    icon_display = f"{icon} " if icon else ""
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {background_color}, {background_color}dd);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 0.5rem 0;
    ">
        <div style="color: white; font-size: 0.9rem; margin-bottom: 0.5rem; opacity: 0.9;">
            {icon_display}{title}
        </div>
        <div style="color: white; font-size: 2rem; font-weight: bold; margin-bottom: 0.25rem;">
            {value}
        </div>
        {f'<div style="color: {"#4ADE80" if delta_color == "normal" else "#F87171"}; font-size: 0.8rem;">{delta}</div>' if delta else ''}
    </div>
    """, unsafe_allow_html=True)

def create_status_indicator(status: str, message: str, icon: str = "") -> None:
    """
    Create a status indicator with appropriate styling
    
    Args:
        status: Status type (success, warning, error, info)
        message: Status message
        icon: Optional emoji icon
    """
    colors = {
        "success": "#10B981",
        "warning": "#F59E0B", 
        "error": "#EF4444",
        "info": "#3B82F6"
    }
    
    color = colors.get(status, "#6B7280")
    icon_display = f"{icon} " if icon else ""
    
    st.markdown(f"""
    <div style="
        background-color: {color}22;
        border-left: 4px solid {color};
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    ">
        <span style="color: {color}; font-weight: 600;">
            {icon_display}{message}
        </span>
    </div>
    """, unsafe_allow_html=True)

def create_progress_ring(percentage: float, size: int = 120, 
                        color: str = "#3B82F6") -> str:
    """
    Create a circular progress ring SVG
    
    Args:
        percentage: Progress percentage (0-100)
        size: Ring size in pixels
        color: Ring color
        
    Returns:
        SVG string for the progress ring
    """
    radius = 45
    circumference = 2 * np.pi * radius
    stroke_dasharray = circumference
    stroke_dashoffset = circumference - (percentage / 100) * circumference
    
    return f"""
    <svg width="{size}" height="{size}" style="transform: rotate(-90deg);">
        <circle cx="60" cy="60" r="{radius}" stroke="#e5e7eb" stroke-width="10" fill="none"/>
        <circle cx="60" cy="60" r="{radius}" stroke="{color}" stroke-width="10" 
                fill="none" stroke-dasharray="{stroke_dasharray}" 
                stroke-dashoffset="{stroke_dashoffset}" stroke-linecap="round"/>
    </svg>
    """

def display_prediction_result(result: Dict[str, Any], model_name: str = "") -> None:
    """
    Display prediction result in a professional format
    
    Args:
        result: Prediction result dictionary
        model_name: Name of the model used
    """
    prediction = result.get('prediction', 'Unknown')
    confidence = result.get('confidence', 0)
    probabilities = result.get('probabilities', {})
    
    # Color coding
    color = "#EF4444" if prediction == 'spam' else "#10B981"
    icon = "🚨" if prediction == 'spam' else "✅"
    
    # Main result card
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}22, {color}11);
        border: 2px solid {color};
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
    ">
        <div style="font-size: 3rem; margin-bottom: 1rem;">{icon}</div>
        <h2 style="color: {color}; margin-bottom: 0.5rem;">
            {prediction.upper()}
        </h2>
        <div style="font-size: 1.2rem; color: #666; margin-bottom: 1rem;">
            {model_name} Model
        </div>
        <div style="font-size: 2rem; font-weight: bold; color: {color};">
            {confidence:.1f}% Confidence
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Probability bars
    if probabilities:
        st.subheader("📊 Probability Breakdown")
        for label, prob in probabilities.items():
            prob_percent = prob * 100
            bar_color = "#EF4444" if label == 'spam' else "#10B981"
            
            st.markdown(f"""
            <div style="margin: 1rem 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="font-weight: 600; color: {bar_color};">{label.title()}</span>
                    <span style="font-weight: 600;">{prob_percent:.1f}%</span>
                </div>
                <div style="
                    background-color: #e5e7eb;
                    border-radius: 10px;
                    height: 20px;
                    overflow: hidden;
                ">
                    <div style="
                        background-color: {bar_color};
                        height: 100%;
                        width: {prob_percent}%;
                        border-radius: 10px;
                        transition: width 0.5s ease;
                    "></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# DATA HANDLING UTILITIES
# =============================================================================

def safe_read_csv(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Safely read CSV file with error handling and encoding detection
    
    Args:
        file_path: Path to CSV file
        **kwargs: Additional arguments for pd.read_csv
        
    Returns:
        DataFrame or empty DataFrame if error
    """
    try:
        # Try common encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding, **kwargs)
                return df
            except UnicodeDecodeError:
                continue
                
        # If all encodings fail, try with error handling
        df = pd.read_csv(file_path, encoding='utf-8', errors='ignore', **kwargs)
        return df
        
    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")
        return pd.DataFrame()

def process_uploaded_file(uploaded_file) -> Tuple[List[str], str]:
    """
    Process uploaded file and extract messages
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Tuple of (messages_list, file_info)
    """
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        file_size = len(uploaded_file.getvalue())
        
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
            
            # Try to find message column
            possible_columns = ['message', 'text', 'content', 'email', 'body', 'msg']
            message_col = None
            
            for col in possible_columns:
                if col in df.columns.str.lower():
                    message_col = df.columns[df.columns.str.lower() == col][0]
                    break
            
            if not message_col:
                # Use first column if no standard column found
                message_col = df.columns[0]
            
            messages = df[message_col].dropna().astype(str).tolist()
            file_info = f"CSV file with {len(messages)} messages (Column: {message_col})"
            
        elif file_extension == 'txt':
            content = uploaded_file.read().decode('utf-8')
            messages = [line.strip() for line in content.split('\n') if line.strip()]
            file_info = f"Text file with {len(messages)} lines"
            
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return [], ""
        
        return messages, file_info
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return [], ""

def create_download_link(data: Union[pd.DataFrame, Dict, str], 
                        filename: str, 
                        file_format: str = "csv") -> str:
    """
    Create a download link for data
    
    Args:
        data: Data to download
        filename: Filename for download
        file_format: Format (csv, json, txt)
        
    Returns:
        Download link HTML
    """
    try:
        if file_format == "csv" and isinstance(data, pd.DataFrame):
            csv = data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            mime_type = "text/csv"
            
        elif file_format == "json":
            json_str = json.dumps(data, indent=2, default=str)
            b64 = base64.b64encode(json_str.encode()).decode()
            mime_type = "application/json"
            
        elif file_format == "txt":
            if isinstance(data, str):
                text = data
            else:
                text = str(data)
            b64 = base64.b64encode(text.encode()).decode()
            mime_type = "text/plain"
            
        else:
            return "Unsupported format"
        
        href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}" style="text-decoration: none;">'
        return href + f'<button style="background: #3B82F6; color: white; padding: 0.5rem 1rem; border: none; border-radius: 5px; cursor: pointer;">📥 Download {file_format.upper()}</button></a>'
        
    except Exception as e:
        return f"Error creating download link: {str(e)}"

# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def create_confidence_chart(predictions: List[Dict[str, Any]], 
                          title: str = "Confidence Distribution") -> go.Figure:
    """
    Create a confidence distribution chart
    
    Args:
        predictions: List of prediction dictionaries
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    confidences = [p.get('confidence', 0) for p in predictions]
    labels = [p.get('prediction', 'unknown') for p in predictions]
    
    fig = go.Figure()
    
    # Separate spam and ham confidences
    spam_confidences = [c for c, l in zip(confidences, labels) if l == 'spam']
    ham_confidences = [c for c, l in zip(confidences, labels) if l == 'ham']
    
    if spam_confidences:
        fig.add_trace(go.Histogram(
            x=spam_confidences,
            name='Spam',
            marker_color='#EF4444',
            opacity=0.7,
            nbinsx=20
        ))
    
    if ham_confidences:
        fig.add_trace(go.Histogram(
            x=ham_confidences,
            name='Ham',
            marker_color='#10B981',
            opacity=0.7,
            nbinsx=20
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Confidence (%)",
        yaxis_title="Count",
        template="plotly_white",
        height=400
    )
    
    return fig

def create_comparison_radar_chart(model1_metrics: Dict, model2_metrics: Dict,
                                model1_name: str = "Model 1", 
                                model2_name: str = "Model 2") -> go.Figure:
    """
    Create a radar chart comparing two models
    
    Args:
        model1_metrics: First model metrics
        model2_metrics: Second model metrics
        model1_name: Name of first model
        model2_name: Name of second model
        
    Returns:
        Plotly figure object
    """
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Speed']
    
    model1_values = [
        model1_metrics.get('accuracy', 0) * 100,
        model1_metrics.get('precision', 0) * 100,
        model1_metrics.get('recall', 0) * 100,
        model1_metrics.get('f1_score', 0) * 100,
        model1_metrics.get('speed_score', 85)  # Default speed score
    ]
    
    model2_values = [
        model2_metrics.get('accuracy', 0) * 100,
        model2_metrics.get('precision', 0) * 100,
        model2_metrics.get('recall', 0) * 100,
        model2_metrics.get('f1_score', 0) * 100,
        model2_metrics.get('speed_score', 90)  # Default speed score
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=model1_values + [model1_values[0]],  # Close the shape
        theta=categories + [categories[0]],
        fill='toself',
        name=model1_name,
        line_color='#3B82F6',
        fillcolor='rgba(59, 130, 246, 0.2)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=model2_values + [model2_values[0]],  # Close the shape
        theta=categories + [categories[0]],
        fill='toself',
        name=model2_name,
        line_color='#EF4444',
        fillcolor='rgba(239, 68, 68, 0.2)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Model Performance Comparison",
        height=500
    )
    
    return fig

def create_time_series_chart(timestamps: List[datetime], 
                           values: List[float],
                           title: str = "Performance Over Time") -> go.Figure:
    """
    Create a time series chart
    
    Args:
        timestamps: List of datetime objects
        values: List of corresponding values
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=values,
        mode='lines+markers',
        line=dict(color='#3B82F6', width=3),
        marker=dict(size=8, color='#1E3A8A'),
        fill='tonexty',
        fillcolor='rgba(59, 130, 246, 0.1)'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Value",
        template="plotly_white",
        height=400
    )
    
    return fig

# =============================================================================
# SESSION STATE MANAGEMENT
# =============================================================================

def initialize_session_state():
    """Initialize Streamlit session state with default values"""
    defaults = {
        'prediction_history': [],
        'total_predictions': 0,
        'spam_count': 0,
        'ham_count': 0,
        'session_start_time': datetime.now(),
        'current_page': 'Home',
        'models_loaded': False,
        'last_batch_results': None,
        'user_preferences': {
            'show_probabilities': True,
            'show_processing_time': True,
            'auto_save_history': True
        }
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def update_prediction_stats(prediction: str):
    """Update prediction statistics in session state"""
    st.session_state.total_predictions += 1
    
    if prediction == 'spam':
        st.session_state.spam_count += 1
    else:
        st.session_state.ham_count += 1

def add_to_history(prediction_data: Dict[str, Any]):
    """Add prediction to history"""
    prediction_data['timestamp'] = datetime.now()
    st.session_state.prediction_history.append(prediction_data)
    
    # Keep only last 100 predictions to manage memory
    if len(st.session_state.prediction_history) > 100:
        st.session_state.prediction_history = st.session_state.prediction_history[-100:]

def get_session_stats() -> Dict[str, Any]:
    """Get current session statistics"""
    session_duration = datetime.now() - st.session_state.session_start_time
    
    return {
        'total_predictions': st.session_state.total_predictions,
        'spam_count': st.session_state.spam_count,
        'ham_count': st.session_state.ham_count,
        'session_duration': session_duration,
        'spam_rate': (st.session_state.spam_count / max(1, st.session_state.total_predictions)) * 100,
        'predictions_per_minute': st.session_state.total_predictions / max(1, session_duration.total_seconds() / 60)
    }

# =============================================================================
# SYSTEM UTILITIES
# =============================================================================

def get_system_info() -> Dict[str, Any]:
    """Get system information and status"""
    return {
        'streamlit_version': st.__version__,
        'current_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'session_id': hashlib.md5(str(st.session_state.get('session_start_time', datetime.now())).encode()).hexdigest()[:8],
        'memory_usage': f"{len(str(st.session_state))} characters in session state"
    }

def log_user_action(action: str, details: Dict[str, Any] = None):
    """Log user actions for analytics (if logging is enabled)"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'action': action,
        'details': details or {},
        'session_id': get_system_info()['session_id']
    }
    
    # In a production environment, you might want to send this to a logging service
    logging.info(f"User Action: {json.dumps(log_entry)}")

def format_elapsed_time(start_time: datetime, end_time: datetime = None) -> str:
    """Format elapsed time in a human-readable format"""
    if end_time is None:
        end_time = datetime.now()
    
    elapsed = end_time - start_time
    
    if elapsed.total_seconds() < 1:
        return f"{elapsed.microseconds // 1000}ms"
    elif elapsed.total_seconds() < 60:
        return f"{elapsed.seconds}.{elapsed.microseconds // 100000}s"
    else:
        minutes = elapsed.seconds // 60
        seconds = elapsed.seconds % 60
        return f"{minutes}m {seconds}s"

def clean_text_for_display(text: str, max_length: int = 100) -> str:
    """Clean and truncate text for display purposes"""
    if not text:
        return "No content"
    
    # Remove excessive whitespace
    cleaned = re.sub(r'\s+', ' ', text.strip())
    
    # Truncate if too long
    if len(cleaned) > max_length:
        return cleaned[:max_length-3] + "..."
    
    return cleaned

# =============================================================================
# SAMPLE DATA GENERATORS
# =============================================================================

def get_sample_messages() -> Dict[str, List[str]]:
    """Get sample messages for testing"""
    return {
        'mixed_sample': [
            "Hey, can we meet for coffee tomorrow?",
            "URGENT! You've won $1000000! Click here NOW!",
            "Thanks for the great meeting today. Looking forward to working together.",
            "Free money! No questions asked! Call 1-800-SCAM-NOW!",
            "The quarterly report is ready for review.",
            "Hot singles in your area want to meet you!",
            "Reminder: Team meeting at 3 PM in conference room B",
            "CONGRATULATIONS! You are our lucky winner!"
        ],
        'spam_heavy': [
            "WIN BIG! Casino offers you FREE $500! No deposit required!",
            "URGENT: Your account will be closed! Verify now at fake-bank.com",
            "Make $5000/week working from home! No experience needed!",
            "FREE iPhone 15! Just pay shipping! Limited time offer!",
            "Your computer is infected! Download our antivirus now!",
            "Hot deal! 90% off luxury watches! Buy now before it's too late!",
            "CONGRATULATIONS! You've been selected for our VIP program!",
            "Lose 30 pounds in 30 days! Try our miracle pill!"
        ],
        'ham_sample': [
            "Hi Sarah, hope you're doing well. Let's catch up soon!",
            "The presentation slides are attached. Please review before Monday.",
            "Thanks for helping me with the project. I really appreciate it.",
            "Reminder: Doctor's appointment tomorrow at 2 PM.",
            "The new restaurant downtown has great reviews. Want to try it?",
            "Can you send me the report when you get a chance?",
            "Happy birthday! Hope you have a wonderful day!",
            "The meeting has been rescheduled to Friday at 10 AM."
        ]
    }

# =============================================================================
# PERFORMANCE MONITORING
# =============================================================================

class PerformanceMonitor:
    """Monitor and track system performance"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.metrics = {}
    
    def start_timer(self):
        """Start performance timing"""
        self.start_time = time.time()
    
    def stop_timer(self):
        """Stop performance timing and return elapsed time"""
        if self.start_time is None:
            return 0
        
        self.end_time = time.time()
        return self.end_time - self.start_time
    
    def add_metric(self, name: str, value: Any):
        """Add a performance metric"""
        self.metrics[name] = value
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        elapsed = self.stop_timer() if self.start_time else 0
        
        return {
            'elapsed_time': elapsed,
            'elapsed_formatted': f"{elapsed:.3f}s" if elapsed > 0 else "N/A",
            'metrics': self.metrics,
            'timestamp': datetime.now().isoformat()
        }

# =============================================================================
# EXPORT UTILITIES
# =============================================================================

def export_results_to_excel(results_df: pd.DataFrame, filename: str = None) -> bytes:
    """Export results to Excel format"""
    if filename is None:
        filename = f"spam_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Results', index=False)
        
        # Add a summary sheet
        summary_data = {
            'Metric': ['Total Messages', 'Spam Count', 'Ham Count', 'Spam Rate'],
            'Value': [
                len(results_df),
                len(results_df[results_df['prediction'] == 'spam']),
                len(results_df[results_df['prediction'] == 'ham']),
                f"{(len(results_df[results_df['prediction'] == 'spam']) / len(results_df) * 100):.1f}%"
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    output.seek(0)
    return output.getvalue()

if __name__ == "__main__":
    # Test helper functions
    print("=== Helper Functions Test ===")
    
    # Test sample data
    samples = get_sample_messages()
    print(f"Sample messages loaded: {sum(len(msgs) for msgs in samples.values())} total")
    
    # Test performance monitor
    monitor = PerformanceMonitor()
    monitor.start_timer()
    time.sleep(0.1)  # Simulate some work
    elapsed = monitor.stop_timer()
    print(f"Performance test: {elapsed:.3f}s")
    
    print("All helper functions ready! 🚀")