"""
Single Prediction Page UI for Spam Detection System
Professional interface for analyzing individual messages with detailed results
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import pandas as pd
from utils.config import Theme, AppConfig, create_header, create_metric_card

def show_prediction_page():
    """Display the single message prediction interface"""
    
    # Page Header
    create_header(
        "🔍 Single Message Analysis",
        "Analyze individual messages with advanced AI models and get detailed confidence scores"
    )
    
    # Check if models are loaded
    if not st.session_state.get('models_loaded', False):
        st.error("❌ Models not loaded. Please return to home page and wait for models to load.", icon="🚨")
        return
    
    predictor = st.session_state.predictor
    
    # Input Section
    st.markdown(f"""
    <div style="background: white; padding: 2rem; border-radius: 12px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin: 1rem 0;">
        <h3 style="color: {Theme.PRIMARY_BLUE}; margin: 0 0 1rem 0; font-weight: 600;">
            📝 Message Input
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Main input area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Message input
        message = st.text_area(
            "Enter the message to analyze:",
            height=200,
            placeholder="Type or paste your email/message here...\n\nExample:\n'Congratulations! You've won $1000! Click here to claim your prize now!'",
            help="Enter any text message you want to analyze for spam detection",
            key="message_input"
        )
        
        # Character count and validation
        if message:
            char_count = len(message)
            word_count = len(message.split())
            
            # Color coding for message length
            if char_count < 10:
                color = Theme.WARNING_ORANGE
                status = "Too short"
            elif char_count > 5000:
                color = Theme.ERROR_RED
                status = "Too long"
            else:
                color = Theme.SUCCESS_GREEN
                status = "Good length"
            
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; margin: 0.5rem 0; 
                        color: {Theme.TEXT_LIGHT}; font-size: 0.9rem;">
                <span>Characters: <strong style="color: {color};">{char_count}</strong> | 
                      Words: <strong>{word_count}</strong></span>
                <span style="color: {color};"><strong>{status}</strong></span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Model selection and options
        st.markdown(f"""
        <h4 style="color: {Theme.PRIMARY_BLUE}; margin: 0 0 1rem 0;">⚙️ Analysis Options</h4>
        """, unsafe_allow_html=True)
        
        # Model selection
        available_models = predictor.get_available_models()
        model_options = {}
        
        for model_key in available_models:
            model_info = predictor.get_model_info(model_key)
            display_name = model_info.get('name', model_key.upper())
            model_options[display_name] = model_key
        
        selected_model_name = st.selectbox(
            "Select Model:",
            options=list(model_options.keys()),
            help="Choose which AI model to use for prediction"
        )
        selected_model = model_options[selected_model_name]
        
        # Analysis mode
        analysis_mode = st.radio(
            "Analysis Mode:",
            ["Single Model", "Model Comparison"],
            help="Choose between single model analysis or comparing both models"
        )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            show_probabilities = st.checkbox("Show probability breakdown", value=True)
            show_processing_time = st.checkbox("Show processing time", value=True)
            auto_save = st.checkbox("Auto-save to history", value=True)
    
    # Prediction Button and Results
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button(
            "🚀 Analyze Message" if analysis_mode == "Single Model" else "🔄 Compare Models",
            use_container_width=True,
            type="primary",
            disabled=not message or len(message.strip()) < 5
        )
    
    # Results Section
    if predict_button and message:
        if analysis_mode == "Single Model":
            show_single_prediction_results(message, selected_model, predictor, show_probabilities, show_processing_time, auto_save)
        else:
            show_comparison_results(message, predictor, show_probabilities, show_processing_time, auto_save)

def show_single_prediction_results(message, model_key, predictor, show_probabilities, show_processing_time, auto_save):
    """Display results for single model prediction"""
    
    try:
        with st.spinner("🤖 Analyzing message..."):
            # Make prediction
            result = predictor.predict_single(message, model_key)
        
        # Save to history if enabled
        if auto_save:
            if 'prediction_history' not in st.session_state:
                st.session_state.prediction_history = []
            st.session_state.prediction_history.append(result)
        
        # Results Header
        st.markdown(f"""
        <div style="background: white; padding: 2rem; border-radius: 12px; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin: 2rem 0;">
            <h3 style="color: {Theme.PRIMARY_BLUE}; margin: 0 0 1rem 0; font-weight: 600;">
                📊 Analysis Results
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Main result display
        prediction_label = result['prediction_label']
        confidence = result['confidence']
        
        # Color coding based on prediction
        if prediction_label == "SPAM":
            result_color = Theme.ERROR_RED
            icon = "🚨"
            bg_gradient = "linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%)"
        else:
            result_color = Theme.SUCCESS_GREEN
            icon = "✅"
            bg_gradient = "linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%)"
        
        # Main prediction card
        st.markdown(f"""
        <div style="background: {bg_gradient}; padding: 2rem; border-radius: 12px; 
                    border-left: 6px solid {result_color}; margin: 1rem 0; text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">{icon}</div>
            <h2 style="color: {result_color}; margin: 0 0 0.5rem 0; font-size: 2.5rem; font-weight: 700;">
                {prediction_label}
            </h2>
            <p style="color: {result_color}; font-size: 1.2rem; font-weight: 600; margin: 0;">
                {confidence:.1%} Confidence
            </p>
            <p style="color: {Theme.TEXT_DARK}; margin: 1rem 0 0 0; font-size: 0.9rem;">
                Model: {result['model_name']}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            create_metric_card(
                "Ham Probability", 
                f"{result['probabilities']['ham']:.1%}",
                "Legitimate message likelihood",
                Theme.SUCCESS_GREEN
            )
        
        with col2:
            create_metric_card(
                "Spam Probability",
                f"{result['probabilities']['spam']:.1%}",
                "Spam message likelihood", 
                Theme.ERROR_RED
            )
        
        with col3:
            if show_processing_time:
                create_metric_card(
                    "Processing Time",
                    f"{result['processing_time']:.3f}s",
                    "Analysis duration"
                )
        
        # Probability visualization
        if show_probabilities:
            st.markdown(f"""
            <h4 style="color: {Theme.PRIMARY_BLUE}; margin: 2rem 0 1rem 0; font-weight: 600;">
                📈 Probability Breakdown
            </h4>
            """, unsafe_allow_html=True)
            
            # Create probability chart
            fig = go.Figure()
            
            categories = ['Ham (Legitimate)', 'Spam (Unwanted)']
            values = [result['probabilities']['ham'], result['probabilities']['spam']]
            colors = [Theme.SUCCESS_GREEN, Theme.ERROR_RED]
            
            fig.add_trace(go.Bar(
                x=categories,
                y=values,
                marker_color=colors,
                text=[f"{v:.1%}" for v in values],
                textposition='auto',
            ))
            
            fig.update_layout(
                title={
                    'text': f"Model Confidence - {result['model_name']}",
                    'x': 0.5,
                    'font': {'size': 16, 'color': Theme.PRIMARY_BLUE}
                },
                yaxis_title="Probability",
                yaxis=dict(tickformat=".0%"),
                showlegend=False,
                height=400,
                margin=dict(t=60, b=40, l=40, r=40)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Message preview
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 12px; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin: 1.5rem 0;">
            <h4 style="color: {Theme.PRIMARY_BLUE}; margin: 0 0 1rem 0;">📄 Analyzed Message</h4>
            <div style="background: {Theme.LIGHT_BLUE}; padding: 1rem; border-radius: 8px; 
                        max-height: 200px; overflow-y: auto;">
                <p style="margin: 0; color: {Theme.TEXT_DARK}; white-space: pre-wrap; font-family: monospace;">
                    {message[:1000]}{'...' if len(message) > 1000 else ''}
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Success message
        st.success("✅ Analysis completed successfully!", icon="🎯")
        
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}", icon="🚨")

def show_comparison_results(message, predictor, show_probabilities, show_processing_time, auto_save):
    """Display results comparing both models"""
    
    try:
        with st.spinner("🔄 Comparing models..."):
            # Get comparison results
            comparison = predictor.compare_models(message)
        
        # Save to history if enabled (save first model result for consistency)
        if auto_save and comparison['models']:
            first_model_key = list(comparison['models'].keys())[0]
            first_result = {
                'message': message,
                'prediction_label': comparison['models'][first_model_key]['prediction_label'],
                'confidence': comparison['models'][first_model_key]['confidence'],
                'model_used': first_model_key,
                'timestamp': comparison['timestamp']
            }
            if 'prediction_history' not in st.session_state:
                st.session_state.prediction_history = []
            st.session_state.prediction_history.append(first_result)
        
        # Results Header
        st.markdown(f"""
        <div style="background: white; padding: 2rem; border-radius: 12px; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin: 2rem 0;">
            <h3 style="color: {Theme.PRIMARY_BLUE}; margin: 0 0 1rem 0; font-weight: 600;">
                🔄 Model Comparison Results
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Agreement indicator
        agreement_color = Theme.SUCCESS_GREEN if comparison['agreement'] else Theme.WARNING_ORANGE
        agreement_text = "Models Agree" if comparison['agreement'] else "Models Disagree"
        agreement_icon = "✅" if comparison['agreement'] else "⚠️"
        
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 12px; 
                    border-left: 4px solid {agreement_color}; margin: 1rem 0; text-align: center;">
            <h4 style="color: {agreement_color}; margin: 0 0 0.5rem 0;">
                {agreement_icon} {agreement_text}
            </h4>
            {f'<p style="color: {Theme.TEXT_LIGHT}; margin: 0;">Confidence difference: {comparison["confidence_difference"]:.1%}</p>' if comparison.get('confidence_difference') else ''}
        </div>
        """, unsafe_allow_html=True)
        
        # Model comparison cards
        cols = st.columns(len(comparison['models']))
        
        for idx, (model_key, result) in enumerate(comparison['models'].items()):
            with cols[idx]:
                prediction_label = result['prediction_label']
                confidence = result['confidence']
                
                # Color coding
                if prediction_label == "SPAM":
                    result_color = Theme.ERROR_RED
                    icon = "🚨"
                else:
                    result_color = Theme.SUCCESS_GREEN
                    icon = "✅"
                
                model_color = Theme.PRIMARY_BLUE if model_key == 'svm' else Theme.SECONDARY_BLUE
                
                st.markdown(f"""
                <div style="background: white; padding: 2rem; border-radius: 12px; 
                            border-left: 4px solid {model_color}; 
                            box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin: 1rem 0; text-align: center;">
                    <h4 style="color: {model_color}; margin: 0 0 1rem 0; font-size: 1.1rem;">
                        {result['model_name']}
                    </h4>
                    <div style="font-size: 2rem; margin: 1rem 0;">{icon}</div>
                    <h3 style="color: {result_color}; margin: 0 0 0.5rem 0; font-size: 1.8rem;">
                        {prediction_label}
                    </h3>
                    <p style="color: {result_color}; font-weight: 600; margin: 0;">
                        {confidence:.1%} Confidence
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        # Detailed probability comparison
        if show_probabilities:
            st.markdown(f"""
            <h4 style="color: {Theme.PRIMARY_BLUE}; margin: 2rem 0 1rem 0; font-weight: 600;">
                📊 Probability Comparison
            </h4>
            """, unsafe_allow_html=True)
            
            # Create comparison chart
            fig = go.Figure()
            
            model_names = [comparison['models'][key]['model_name'] for key in comparison['models']]
            ham_probs = [comparison['models'][key]['probabilities']['ham'] for key in comparison['models']]
            spam_probs = [comparison['models'][key]['probabilities']['spam'] for key in comparison['models']]
            
            fig.add_trace(go.Bar(
                name='Ham (Legitimate)',
                x=model_names,
                y=ham_probs,
                marker_color=Theme.SUCCESS_GREEN,
                text=[f"{v:.1%}" for v in ham_probs],
                textposition='auto'
            ))
            
            fig.add_trace(go.Bar(
                name='Spam (Unwanted)',
                x=model_names,
                y=spam_probs,
                marker_color=Theme.ERROR_RED,
                text=[f"{v:.1%}" for v in spam_probs],
                textposition='auto'
            ))
            
            fig.update_layout(
                title={
                    'text': "Model Probability Comparison",
                    'x': 0.5,
                    'font': {'size': 16, 'color': Theme.PRIMARY_BLUE}
                },
                yaxis_title="Probability",
                yaxis=dict(tickformat=".0%"),
                barmode='group',
                height=400,
                margin=dict(t=60, b=40, l=40, r=40)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Success message
        st.success("✅ Model comparison completed successfully!", icon="🔄")
        
    except Exception as e:
        st.error(f"❌ Error during model comparison: {str(e)}", icon="🚨")

# Example messages for quick testing
def show_example_messages():
    """Show example messages for quick testing"""
    st.markdown(f"""
    <div style="background: {Theme.LIGHT_BLUE}; padding: 1.5rem; border-radius: 12px; margin: 1rem 0;">
        <h4 style="color: {Theme.PRIMARY_BLUE}; margin: 0 0 1rem 0;">💡 Try These Examples</h4>
        <p style="color: {Theme.TEXT_DARK}; margin: 0 0 1rem 0;">Click on any example to test it:</p>
    </div>
    """, unsafe_allow_html=True)
    
    examples = [
        {
            "type": "SPAM",
            "text": "CONGRATULATIONS! You've won $10,000! Click here NOW to claim your prize before it expires!",
            "description": "Typical lottery scam"
        },
        {
            "type": "HAM", 
            "text": "Hi John, can we reschedule our meeting for tomorrow at 3 PM? Let me know if that works for you.",
            "description": "Legitimate business email"
        },
        {
            "type": "SPAM",
            "text": "URGENT: Your account will be suspended! Click this link immediately to verify your information.",
            "description": "Phishing attempt"
        }
    ]
    
    for i, example in enumerate(examples):
        color = Theme.ERROR_RED if example["type"] == "SPAM" else Theme.SUCCESS_GREEN
        
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"""
            <div style="background: white; padding: 1rem; border-radius: 8px; 
                        border-left: 3px solid {color}; margin: 0.5rem 0;">
                <strong style="color: {color};">{example['type']} Example:</strong> {example['description']}<br>
                <em style="color: {Theme.TEXT_LIGHT};">"{example['text'][:100]}..."</em>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if st.button(f"Test #{i+1}", key=f"example_{i}"):
                st.session_state.message_input = example['text']
                st.rerun()