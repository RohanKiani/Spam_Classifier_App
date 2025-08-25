"""
Batch Analysis Page UI for Spam Detection System
Professional interface for processing multiple messages and comprehensive model comparison
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import io
import time
from utils.config import Theme, AppConfig, create_header, create_metric_card

def show_batch_analysis_page():
    """Display the batch processing and analysis interface"""
    
    # Page Header
    create_header(
        "📊 Batch Analysis & Model Comparison",
        "Process multiple messages simultaneously and compare model performance with advanced analytics"
    )
    
    # Check if models are loaded
    if not st.session_state.get('models_loaded', False):
        st.error("❌ Models not loaded. Please return to home page and wait for models to load.", icon="🚨")
        return
    
    predictor = st.session_state.predictor
    
    # Input Methods Section
    st.markdown(f"""
    <div style="background: white; padding: 2rem; border-radius: 12px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin: 1rem 0;">
        <h3 style="color: {Theme.PRIMARY_BLUE}; margin: 0 0 1rem 0; font-weight: 600;">
            📥 Message Input Methods
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Input method tabs
    tab1, tab2, tab3 = st.tabs(["📝 Manual Input", "📄 File Upload", "🔗 Sample Data"])
    
    messages_to_analyze = []
    
    with tab1:
        show_manual_input_section(messages_to_analyze)
    
    with tab2:
        show_file_upload_section(messages_to_analyze)
    
    with tab3:
        show_sample_data_section(messages_to_analyze)
    
    # Analysis Configuration
    if messages_to_analyze:
        st.markdown(f"""
        <div style="background: white; padding: 2rem; border-radius: 12px; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin: 2rem 0;">
            <h3 style="color: {Theme.PRIMARY_BLUE}; margin: 0 0 1rem 0; font-weight: 600;">
                ⚙️ Analysis Configuration
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        show_analysis_configuration(messages_to_analyze, predictor)

def show_manual_input_section(messages_to_analyze):
    """Manual input section for typing/pasting messages"""
    
    st.markdown(f"""
    <p style="color: {Theme.TEXT_DARK}; margin-bottom: 1rem;">
        Enter multiple messages (one per line) for batch analysis:
    </p>
    """, unsafe_allow_html=True)
    
    manual_input = st.text_area(
        "Messages (one per line):",
        height=200,
        placeholder="Enter messages here, one per line:\n\nWin $1000 now! Click here!\nHi, can we meet tomorrow at 3 PM?\nUrgent: Your account needs verification\nThanks for the great meeting today",
        help="Each line will be treated as a separate message for analysis"
    )
    
    if manual_input:
        lines = [line.strip() for line in manual_input.split('\n') if line.strip()]
        valid_messages = [msg for msg in lines if len(msg) >= 5]
        
        if valid_messages:
            messages_to_analyze.extend(valid_messages)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Lines", len(lines))
            with col2:
                st.metric("Valid Messages", len(valid_messages))
            with col3:
                invalid_count = len(lines) - len(valid_messages)
                st.metric("Invalid/Short", invalid_count)
            
            if invalid_count > 0:
                st.warning(f"⚠️ {invalid_count} messages were too short (minimum 5 characters)", icon="📏")

def show_file_upload_section(messages_to_analyze):
    """File upload section for CSV/TXT files"""
    
    st.markdown(f"""
    <p style="color: {Theme.TEXT_DARK}; margin-bottom: 1rem;">
        Upload a file containing messages for batch processing:
    </p>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['txt', 'csv'],
        help="Upload a .txt file (one message per line) or .csv file (with a 'message' or 'text' column)"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.type == "text/plain":
                # Handle TXT files
                content = str(uploaded_file.read(), "utf-8")
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                valid_messages = [msg for msg in lines if len(msg) >= 5]
                
                if valid_messages:
                    messages_to_analyze.extend(valid_messages)
                    st.success(f"✅ Loaded {len(valid_messages)} messages from {uploaded_file.name}")
                else:
                    st.error("❌ No valid messages found in the file")
                    
            elif uploaded_file.type == "text/csv":
                # Handle CSV files
                df = pd.read_csv(uploaded_file)
                
                # Try to find message column
                possible_columns = ['message', 'text', 'content', 'email', 'body']
                message_column = None
                
                for col in possible_columns:
                    if col.lower() in [c.lower() for c in df.columns]:
                        message_column = next(c for c in df.columns if c.lower() == col.lower())
                        break
                
                if message_column:
                    messages = df[message_column].astype(str).tolist()
                    valid_messages = [msg for msg in messages if len(str(msg).strip()) >= 5]
                    
                    if valid_messages:
                        messages_to_analyze.extend(valid_messages)
                        st.success(f"✅ Loaded {len(valid_messages)} messages from column '{message_column}'")
                        
                        # Show CSV preview
                        with st.expander("📄 File Preview"):
                            st.dataframe(df.head(10), use_container_width=True)
                    else:
                        st.error("❌ No valid messages found in the CSV file")
                else:
                    st.error(f"❌ Could not find message column. Available columns: {', '.join(df.columns)}")
                    st.info("💡 Make sure your CSV has a column named: 'message', 'text', 'content', 'email', or 'body'")
                    
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")

def show_sample_data_section(messages_to_analyze):
    """Sample data section with pre-defined test messages"""
    
    st.markdown(f"""
    <p style="color: {Theme.TEXT_DARK}; margin-bottom: 1rem;">
        Use our curated sample dataset for testing and demonstration:
    </p>
    """, unsafe_allow_html=True)
    
    sample_datasets = {
        "Mixed Sample (10 messages)": [
            "Congratulations! You've won $5000! Click here to claim now!",
            "Hi team, please review the quarterly report by Friday.",
            "URGENT: Your account will be suspended unless you verify immediately!",
            "Thanks for the great presentation yesterday. Let's discuss next steps.",
            "Get rich quick! Make $500/day working from home!",
            "Meeting reminder: Project review at 2 PM in conference room B.",
            "Free iPhone! Limited time offer! Click now!",
            "Could you please send me the updated budget spreadsheet?",
            "WINNER! You are our lucky customer! Claim your prize!",
            "Don't forget about our team lunch this Friday at noon."
        ],
        "Spam Heavy (15 messages)": [
            "Win big money now! Click here for instant cash!",
            "Free vacation to Hawaii! All expenses paid!",
            "Your computer is infected! Download our antivirus now!",
            "Congratulations winner! Claim your lottery prize!",
            "Make money fast! Work from home opportunity!",
            "Free iPhone 15! Limited time offer!",
            "URGENT: Bank account verification required!",
            "Get rich quick with our secret system!",
            "Free gift card waiting for you! Claim now!",
            "Weight loss miracle! Lose 20 pounds in 10 days!",
            "Investment opportunity! Double your money!",
            "Free credit report! No strings attached!",
            "Pharmacy online! Cheap medications!",
            "Meet singles in your area! Free registration!",
            "Debt consolidation! Lower your payments now!"
        ],
        "Ham Heavy (12 messages)": [
            "Hi John, can we reschedule our meeting for tomorrow?",
            "Thanks for sending the documents. I'll review them today.",
            "Reminder: Team standup at 9 AM tomorrow morning.",
            "Great job on the presentation! The client was impressed.",
            "Could you please update the project timeline?",
            "Let's grab coffee and discuss the new requirements.",
            "The quarterly results look promising. Well done team!",
            "Please find the attached invoice for last month's services.",
            "Happy birthday! Hope you have a wonderful day.",
            "The system maintenance is scheduled for this weekend.",
            "Thanks for your help with the technical issue yesterday.",
            "Looking forward to working with you on this project."
        ]
    }
    
    selected_dataset = st.selectbox(
        "Choose a sample dataset:",
        options=list(sample_datasets.keys()),
        help="Select a pre-defined dataset for testing the batch analysis features"
    )
    
    if st.button("📊 Load Sample Dataset", use_container_width=True):
        sample_messages = sample_datasets[selected_dataset]
        messages_to_analyze.extend(sample_messages)
        st.success(f"✅ Loaded {len(sample_messages)} sample messages")
        
        # Show preview
        with st.expander("👀 Dataset Preview"):
            for i, msg in enumerate(sample_messages[:5], 1):
                st.write(f"**{i}.** {msg}")
            if len(sample_messages) > 5:
                st.write(f"... and {len(sample_messages) - 5} more messages")

def show_analysis_configuration(messages_to_analyze, predictor):
    """Analysis configuration and execution section"""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Analysis options
        st.markdown("### 🔧 Analysis Options")
        
        analysis_type = st.radio(
            "Analysis Type:",
            ["Single Model Analysis", "Model Comparison", "Both Models Separately"],
            help="Choose how you want to analyze the messages"
        )
        
        if analysis_type == "Single Model Analysis":
            available_models = predictor.get_available_models()
            model_options = {}
            
            for model_key in available_models:
                model_info = predictor.get_model_info(model_key)
                display_name = model_info.get('name', model_key.upper())
                model_options[display_name] = model_key
            
            selected_model_name = st.selectbox(
                "Select Model:",
                options=list(model_options.keys())
            )
            selected_model = model_options[selected_model_name]
        
        # Output options
        col_a, col_b = st.columns(2)
        with col_a:
            show_detailed_results = st.checkbox("Show detailed results", value=True)
            include_charts = st.checkbox("Include visualizations", value=True)
        with col_b:
            export_results = st.checkbox("Enable export", value=True)
            save_to_history = st.checkbox("Save to session history", value=True)
    
    with col2:
        # Summary info
        st.markdown("### 📋 Batch Summary")
        
        create_metric_card("Messages Ready", str(len(messages_to_analyze)), "For processing")
        
        if len(messages_to_analyze) > 100:
            st.warning("⚠️ Large batch detected. Processing may take longer.", icon="⏱️")
        
        # Estimated processing time
        estimated_time = len(messages_to_analyze) * 0.01  # Rough estimate
        st.info(f"⏱️ Estimated processing time: ~{estimated_time:.1f} seconds")
    
    # Process button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(
            f"🚀 Process {len(messages_to_analyze)} Messages",
            use_container_width=True,
            type="primary",
            disabled=len(messages_to_analyze) == 0
        ):
            process_batch_analysis(
                messages_to_analyze, 
                predictor, 
                analysis_type, 
                selected_model if analysis_type == "Single Model Analysis" else None,
                show_detailed_results,
                include_charts,
                export_results,
                save_to_history
            )

def process_batch_analysis(messages, predictor, analysis_type, selected_model, show_detailed, include_charts, export_results, save_to_history):
    """Process the batch analysis and display results"""
    
    try:
        start_time = time.time()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results_data = {}
        
        if analysis_type == "Single Model Analysis":
            status_text.text(f"🤖 Processing with {predictor.get_model_info(selected_model)['name']}...")
            
            # Process batch
            df_results = predictor.predict_batch(messages, selected_model)
            results_data[selected_model] = df_results
            progress_bar.progress(1.0)
            
        elif analysis_type == "Model Comparison":
            available_models = predictor.get_available_models()
            
            for i, model_key in enumerate(available_models):
                status_text.text(f"🤖 Processing with {predictor.get_model_info(model_key)['name']}...")
                df_results = predictor.predict_batch(messages, model_key)
                results_data[model_key] = df_results
                progress_bar.progress((i + 1) / len(available_models))
        
        elif analysis_type == "Both Models Separately":
            available_models = predictor.get_available_models()
            
            for i, model_key in enumerate(available_models):
                status_text.text(f"🤖 Processing with {predictor.get_model_info(model_key)['name']}...")
                df_results = predictor.predict_batch(messages, model_key)
                results_data[model_key] = df_results
                progress_bar.progress((i + 1) / len(available_models))
        
        total_time = time.time() - start_time
        status_text.empty()
        progress_bar.empty()
        
        # Display results
        display_batch_results(results_data, analysis_type, total_time, show_detailed, include_charts, export_results, save_to_history)
        
    except Exception as e:
        st.error(f"❌ Error during batch processing: {str(e)}", icon="🚨")

def display_batch_results(results_data, analysis_type, total_time, show_detailed, include_charts, export_results, save_to_history):
    """Display comprehensive batch analysis results"""
    
    # Results header
    st.markdown(f"""
    <div style="background: white; padding: 2rem; border-radius: 12px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin: 2rem 0;">
        <h2 style="color: {Theme.PRIMARY_BLUE}; margin: 0 0 1rem 0; font-weight: 600;">
            📊 Batch Analysis Results
        </h2>
        <p style="color: {Theme.TEXT_LIGHT}; margin: 0;">
            Processing completed in {total_time:.2f} seconds
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Overall statistics
    show_overall_statistics(results_data)
    
    if analysis_type == "Model Comparison":
        show_model_comparison_analysis(results_data)
    else:
        # Show individual model results
        for model_key, df_results in results_data.items():
            show_individual_model_results(model_key, df_results, show_detailed, include_charts)
    
    # Export functionality
    if export_results:
        show_export_options(results_data)
    
    # Save to history
    if save_to_history:
        save_results_to_history(results_data)

def show_overall_statistics(results_data):
    """Display overall batch statistics"""
    
    st.markdown(f"""
    <h3 style="color: {Theme.PRIMARY_BLUE}; margin: 2rem 0 1rem 0; font-weight: 600;">
        📈 Overall Statistics
    </h3>
    """, unsafe_allow_html=True)
    
    # Get first dataset for overall stats
    first_df = list(results_data.values())[0]
    total_messages = len(first_df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_metric_card("Total Messages", str(total_messages), "Processed successfully")
    
    with col2:
        spam_count = len(first_df[first_df['prediction_label'] == 'SPAM'])
        create_metric_card("Spam Detected", str(spam_count), f"{spam_count/total_messages*100:.1f}% of total", Theme.ERROR_RED)
    
    with col3:
        ham_count = total_messages - spam_count
        create_metric_card("Ham Messages", str(ham_count), f"{ham_count/total_messages*100:.1f}% of total", Theme.SUCCESS_GREEN)
    
    with col4:
        avg_confidence = first_df['confidence'].mean()
        create_metric_card("Avg Confidence", f"{avg_confidence:.1%}", "Overall certainty")

def show_model_comparison_analysis(results_data):
    """Show detailed model comparison analysis"""
    
    st.markdown(f"""
    <h3 style="color: {Theme.PRIMARY_BLUE}; margin: 2rem 0 1rem 0; font-weight: 600;">
        🔄 Model Comparison Analysis
    </h3>
    """, unsafe_allow_html=True)
    
    if len(results_data) < 2:
        st.warning("⚠️ Need at least 2 models for comparison analysis")
        return
    
    model_keys = list(results_data.keys())
    df1, df2 = results_data[model_keys[0]], results_data[model_keys[1]]
    
    # Agreement analysis
    agreements = (df1['prediction_label'] == df2['prediction_label']).sum()
    total = len(df1)
    agreement_rate = agreements / total * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Agreement metrics
        agreement_color = Theme.SUCCESS_GREEN if agreement_rate > 80 else Theme.WARNING_ORANGE if agreement_rate > 60 else Theme.ERROR_RED
        
        st.markdown(f"""
        <div style="background: white; padding: 2rem; border-radius: 12px; 
                    border-left: 4px solid {agreement_color}; margin: 1rem 0; text-align: center;">
            <h2 style="color: {agreement_color}; margin: 0; font-size: 2.5rem;">{agreement_rate:.1f}%</h2>
            <p style="color: {agreement_color}; font-weight: 600; margin: 0.5rem 0;">Model Agreement</p>
            <p style="color: {Theme.TEXT_LIGHT}; margin: 0; font-size: 0.9rem;">
                {agreements} out of {total} predictions match
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Confidence difference
        conf_diff = abs(df1['confidence'] - df2['confidence']).mean()
        
        st.markdown(f"""
        <div style="background: white; padding: 2rem; border-radius: 12px; 
                    border-left: 4px solid {Theme.PRIMARY_BLUE}; margin: 1rem 0; text-align: center;">
            <h2 style="color: {Theme.PRIMARY_BLUE}; margin: 0; font-size: 2.5rem;">{conf_diff:.1%}</h2>
            <p style="color: {Theme.PRIMARY_BLUE}; font-weight: 600; margin: 0.5rem 0;">Avg Confidence Diff</p>
            <p style="color: {Theme.TEXT_LIGHT}; margin: 0; font-size: 0.9rem;">
                Average difference in certainty
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed comparison chart
    fig = go.Figure()
    
    model_names = [st.session_state.predictor.get_model_info(key)['name'] for key in model_keys]
    
    # Spam counts for each model
    spam_counts = [len(df[df['prediction_label'] == 'SPAM']) for df in results_data.values()]
    ham_counts = [len(df[df['prediction_label'] == 'HAM']) for df in results_data.values()]
    
    fig.add_trace(go.Bar(
        name='Ham',
        x=model_names,
        y=ham_counts,
        marker_color=Theme.SUCCESS_GREEN
    ))
    
    fig.add_trace(go.Bar(
        name='Spam',
        x=model_names,
        y=spam_counts,
        marker_color=Theme.ERROR_RED
    ))
    
    fig.update_layout(
        title='Model Prediction Comparison',
        xaxis_title='Model',
        yaxis_title='Number of Messages',
        barmode='stack',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_individual_model_results(model_key, df_results, show_detailed, include_charts):
    """Show results for an individual model"""
    
    model_info = st.session_state.predictor.get_model_info(model_key)
    model_name = model_info.get('name', model_key.upper())
    
    st.markdown(f"""
    <h3 style="color: {Theme.PRIMARY_BLUE}; margin: 2rem 0 1rem 0; font-weight: 600;">
        🤖 {model_name} Results
    </h3>
    """, unsafe_allow_html=True)
    
    if show_detailed:
        # Show detailed dataframe
        st.dataframe(
            df_results[['message', 'prediction_label', 'confidence', 'ham_probability', 'spam_probability']],
            use_container_width=True,
            hide_index=True
        )
    
    if include_charts:
        col1, col2 = st.columns(2)
        
        with col1:
            # Prediction distribution
            spam_count = len(df_results[df_results['prediction_label'] == 'SPAM'])
            ham_count = len(df_results) - spam_count
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Ham', 'Spam'],
                values=[ham_count, spam_count],
                hole=.3,
                marker_colors=[Theme.SUCCESS_GREEN, Theme.ERROR_RED]
            )])
            
            fig_pie.update_layout(
                title=f'{model_name} - Classification Distribution',
                height=300
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Confidence distribution
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=df_results['confidence'],
                nbinsx=20,
                marker_color=Theme.SECONDARY_BLUE,
                opacity=0.7
            ))
            
            fig_hist.update_layout(
                title=f'{model_name} - Confidence Distribution',
                xaxis_title='Confidence Score',
                yaxis_title='Frequency',
                height=300
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)

def show_export_options(results_data):
    """Show export options for batch results"""
    
    st.markdown(f"""
    <h3 style="color: {Theme.PRIMARY_BLUE}; margin: 2rem 0 1rem 0; font-weight: 600;">
        💾 Export Results
    </h3>
    """, unsafe_allow_html=True)
    
    export_format = st.radio(
        "Export Format:",
        ["CSV", "Excel", "JSON"],
        horizontal=True
    )
    
    col1, col2, col3 = st.columns(3)
    
    for i, (model_key, df_results) in enumerate(results_data.items()):
        model_info = st.session_state.predictor.get_model_info(model_key)
        model_name = model_info.get('name', model_key.upper())
        
        with [col1, col2, col3][i % 3]:
            if export_format == "CSV":
                csv = df_results.to_csv(index=False)
                st.download_button(
                    label=f"📊 Download {model_name} CSV",
                    data=csv,
                    file_name=f"spam_detection_{model_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            elif export_format == "Excel":
                buffer = io.BytesIO()
                df_results.to_excel(buffer, index=False)
                st.download_button(
                    label=f"📊 Download {model_name} Excel",
                    data=buffer.getvalue(),
                    file_name=f"spam_detection_{model_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            elif export_format == "JSON":
                json_data = df_results.to_json(orient='records', indent=2)
                st.download_button(
                    label=f"📊 Download {model_name} JSON",
                    data=json_data,
                    file_name=f"spam_detection_{model_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

def save_results_to_history(results_data):
    """Save batch results to session history"""
    
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    # Save summary to history (use first model's results)
    first_model_key = list(results_data.keys())[0]
    first_df = results_data[first_model_key]
    
    batch_summary = {
        'timestamp': datetime.now().isoformat(),
        'type': 'batch_analysis',
        'total_messages': len(first_df),
        'spam_count': len(first_df[first_df['prediction_label'] == 'SPAM']),
        'ham_count': len(first_df[first_df['prediction_label'] == 'HAM']),
        'models_used': list(results_data.keys())
    }
    
    st.session_state.prediction_history.append(batch_summary)
    st.success("✅ Results saved to session history!", icon="💾")