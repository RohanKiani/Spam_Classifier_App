"""
Home Page UI for Spam Detection System
Professional welcome interface with system overview and quick stats
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from utils.config import Theme, AppConfig, create_header, create_metric_card

def show_home_page():
    """Display the professional home page with system overview"""
    
    # Main Header
    create_header(
        AppConfig.APP_NAME,
        "Professional AI-powered email classification system with advanced machine learning models"
    )
    
    # Welcome section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div style="background: white; padding: 2rem; border-radius: 12px; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin: 1rem 0;">
            <h3 style="color: {Theme.PRIMARY_BLUE}; margin-bottom: 1rem; font-weight: 600;">
                🚀 Welcome to Advanced Spam Detection
            </h3>
            <p style="color: {Theme.TEXT_DARK}; font-size: 1.1rem; line-height: 1.6; margin-bottom: 1rem;">
                Our state-of-the-art machine learning system uses <strong>dual-model architecture</strong> 
                to provide highly accurate spam detection with confidence scoring.
            </p>
            <div style="background: {Theme.LIGHT_BLUE}; padding: 1.5rem; border-radius: 8px; 
                        border-left: 4px solid {Theme.PRIMARY_BLUE}; margin: 1rem 0;">
                <h4 style="color: {Theme.PRIMARY_BLUE}; margin: 0 0 0.5rem 0;">✨ Key Features</h4>
                <ul style="color: {Theme.TEXT_DARK}; margin: 0; padding-left: 1.2rem;">
                    <li><strong>Dual Model System:</strong> SVM + Logistic Regression for maximum accuracy</li>
                    <li><strong>Probability Scoring:</strong> Get confidence levels for each prediction</li>
                    <li><strong>Batch Processing:</strong> Analyze multiple messages simultaneously</li>
                    <li><strong>Real-time Analysis:</strong> Instant results with professional reporting</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # System status card
        if st.session_state.get('models_loaded', False):
            predictor = st.session_state.predictor
            stats = predictor.get_model_stats()
            
            st.markdown(f"""
            <div style="background: {Theme.BLUE_GRADIENT}; padding: 2rem; border-radius: 12px; 
                        text-align: center; margin: 1rem 0;">
                <h3 style="color: white; margin: 0 0 1rem 0;">🤖 System Status</h3>
                <div style="background: rgba(255,255,255,0.1); border-radius: 8px; padding: 1rem; margin: 1rem 0;">
                    <h2 style="color: white; margin: 0; font-size: 2.5rem;">{stats['total_models']}</h2>
                    <p style="color: rgba(255,255,255,0.8); margin: 0;">Models Active</p>
                </div>
                <p style="color: rgba(255,255,255,0.9); margin: 0; font-size: 0.9rem;">
                    ✅ All systems operational
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick action buttons
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 12px; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin: 1rem 0;">
            <h4 style="color: {Theme.PRIMARY_BLUE}; margin: 0 0 1rem 0; text-align: center;">
                🎯 Quick Actions
            </h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Action buttons
        if st.button("🔍 Single Prediction", use_container_width=True, type="primary"):
            st.session_state.current_page = "Prediction"
            st.rerun()
        
        if st.button("📊 Batch Analysis", use_container_width=True):
            st.session_state.current_page = "Batch Analysis" 
            st.rerun()

    # Model Information Cards
    st.markdown(f"""
    <h3 style="color: {Theme.PRIMARY_BLUE}; margin: 2rem 0 1rem 0; font-weight: 600;">
        🤖 Available Models
    </h3>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # SVM Model Card
        st.markdown(f"""
        <div class="model-card svm" style="background: white; padding: 2rem; border-radius: 12px; 
                    border-left: 4px solid {Theme.PRIMARY_BLUE}; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin: 1rem 0;
                    transition: transform 0.3s ease;">
            <h4 style="color: {Theme.PRIMARY_BLUE}; margin: 0 0 1rem 0; display: flex; align-items: center;">
                <span style="font-size: 1.5rem; margin-right: 0.5rem;">⚡</span>
                Support Vector Machine
            </h4>
            <p style="color: {Theme.TEXT_DARK}; margin-bottom: 1rem;">
                <strong>Calibrated SVM</strong> with TF-IDF vectorization for high-precision classification.
            </p>
            <div style="background: {Theme.LIGHT_BLUE}; padding: 1rem; border-radius: 8px;">
                <p style="margin: 0; color: {Theme.TEXT_DARK};"><strong>Features:</strong></p>
                <ul style="margin: 0.5rem 0 0 0; color: {Theme.TEXT_LIGHT}; font-size: 0.9rem;">
                    <li>Probability calibration for confidence scores</li>
                    <li>Excellent performance on text classification</li>
                    <li>Robust against overfitting</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Logistic Regression Model Card
        st.markdown(f"""
        <div class="model-card lr" style="background: white; padding: 2rem; border-radius: 12px; 
                    border-left: 4px solid {Theme.SECONDARY_BLUE}; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin: 1rem 0;
                    transition: transform 0.3s ease;">
            <h4 style="color: {Theme.SECONDARY_BLUE}; margin: 0 0 1rem 0; display: flex; align-items: center;">
                <span style="font-size: 1.5rem; margin-right: 0.5rem;">🎯</span>
                Logistic Regression
            </h4>
            <p style="color: {Theme.TEXT_DARK}; margin-bottom: 1rem;">
                <strong>Linear model</strong> with built-in probability estimates for interpretable results.
            </p>
            <div style="background: {Theme.LIGHT_BLUE}; padding: 1rem; border-radius: 8px;">
                <p style="margin: 0; color: {Theme.TEXT_DARK};"><strong>Features:</strong></p>
                <ul style="margin: 0.5rem 0 0 0; color: {Theme.TEXT_LIGHT}; font-size: 0.9rem;">
                    <li>Natural probability outputs</li>
                    <li>Fast prediction speed</li>
                    <li>Highly interpretable coefficients</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Session Statistics
    if st.session_state.prediction_history:
        st.markdown(f"""
        <h3 style="color: {Theme.PRIMARY_BLUE}; margin: 2rem 0 1rem 0; font-weight: 600;">
            📊 Session Analytics
        </h3>
        """, unsafe_allow_html=True)
        
        # Calculate statistics
        history = st.session_state.prediction_history
        total_predictions = len(history)
        spam_count = sum(1 for p in history if p.get('prediction_label') == 'SPAM')
        ham_count = total_predictions - spam_count
        spam_rate = (spam_count / total_predictions * 100) if total_predictions > 0 else 0
        
        # Statistics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            create_metric_card("Total Messages", str(total_predictions), "Analyzed this session")
        
        with col2:
            create_metric_card("Spam Detected", str(spam_count), f"{spam_rate:.1f}% of total", Theme.ERROR_RED)
        
        with col3:
            create_metric_card("Ham Messages", str(ham_count), f"{100-spam_rate:.1f}% of total", Theme.SUCCESS_GREEN)
        
        with col4:
            avg_confidence = np.mean([p.get('confidence', 0) for p in history])
            create_metric_card("Avg Confidence", f"{avg_confidence:.1%}", "Model certainty")
        
        # Prediction distribution chart
        if total_predictions > 0:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create pie chart for spam/ham distribution
                fig = go.Figure(data=[go.Pie(
                    labels=['Ham', 'Spam'],
                    values=[ham_count, spam_count],
                    hole=.3,
                    marker_colors=[Theme.SUCCESS_GREEN, Theme.ERROR_RED]
                )])
                
                fig.update_layout(
                    title={
                        'text': "Message Classification Distribution",
                        'x': 0.5,
                        'font': {'size': 16, 'color': Theme.PRIMARY_BLUE}
                    },
                    showlegend=True,
                    height=300,
                    margin=dict(t=50, b=20, l=20, r=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Confidence distribution
                confidences = [p.get('confidence', 0) for p in history[-10:]]  # Last 10 predictions
                
                if confidences:
                    fig_confidence = go.Figure()
                    fig_confidence.add_trace(go.Bar(
                        x=list(range(1, len(confidences) + 1)),
                        y=confidences,
                        marker_color=Theme.SECONDARY_BLUE,
                        name='Confidence'
                    ))
                    
                    fig_confidence.update_layout(
                        title={
                            'text': "Recent Prediction Confidence",
                            'x': 0.5,
                            'font': {'size': 14, 'color': Theme.PRIMARY_BLUE}
                        },
                        xaxis_title="Prediction #",
                        yaxis_title="Confidence",
                        height=300,
                        margin=dict(t=50, b=40, l=40, r=20)
                    )
                    
                    st.plotly_chart(fig_confidence, use_container_width=True)

    # How to Use Section
    st.markdown(f"""
    <h3 style="color: {Theme.PRIMARY_BLUE}; margin: 2rem 0 1rem 0; font-weight: 600;">
        📋 How to Use This System
    </h3>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 12px; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05); text-align: center;">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">🔍</div>
            <h4 style="color: {Theme.PRIMARY_BLUE}; margin-bottom: 1rem;">Single Prediction</h4>
            <p style="color: {Theme.TEXT_DARK}; font-size: 0.9rem;">
                Analyze individual messages with detailed confidence scores and model comparisons.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 12px; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05); text-align: center;">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">📊</div>
            <h4 style="color: {Theme.PRIMARY_BLUE}; margin-bottom: 1rem;">Batch Analysis</h4>
            <p style="color: {Theme.TEXT_DARK}; font-size: 0.9rem;">
                Process multiple messages at once and compare model performance side-by-side.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 12px; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05); text-align: center;">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">⚡</div>
            <h4 style="color: {Theme.PRIMARY_BLUE}; margin-bottom: 1rem;">Real-time Results</h4>
            <p style="color: {Theme.TEXT_DARK}; font-size: 0.9rem;">
                Get instant predictions with professional reporting and exportable results.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Footer with additional info
    st.markdown(f"""
    <div style="background: {Theme.LIGHT_BLUE}; padding: 2rem; border-radius: 12px; 
                margin: 2rem 0; text-align: center;">
        <h4 style="color: {Theme.PRIMARY_BLUE}; margin-bottom: 1rem;">🎯 Ready to Get Started?</h4>
        <p style="color: {Theme.TEXT_DARK}; margin-bottom: 1.5rem;">
            Choose an option from the sidebar to begin analyzing your messages with our advanced AI system.
        </p>
        <p style="color: {Theme.TEXT_LIGHT}; font-size: 0.9rem; margin: 0;">
            <strong>Tip:</strong> Use the Single Prediction page for quick tests, 
            or the Batch Analysis page for comprehensive email processing.
        </p>
    </div>
    """, unsafe_allow_html=True)