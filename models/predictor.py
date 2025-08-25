"""
Spam Detection Model Predictor
Handles model loading, caching, and prediction logic for both SVM and Logistic Regression models
"""
import os
import logging
import time
import pandas as pd
import streamlit as st
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelPredictor:
    """
    Professional spam detection predictor class
    Handles multiple models with caching and error handling
    """
    
    def __init__(self):
        self.model_info = {
            'svm': {
                'name': 'Support Vector Machine (Calibrated)',
                'filename': 'linear_svm_tfidf_calibrated.pkl',
                'color': '#1E3A8A',
                'description': 'Calibrated SVM with probability estimates'
            },
            'lr': {
                'name': 'Logistic Regression',
                'filename': 'logistic_regression_tfidf.pkl', 
                'color': '#3B82F6',
                'description': 'Linear model with built-in probabilities'
            }
        }
        self.models = self.load_models()  # <-- Correct assignment
    
    @st.cache_resource
    def load_models(_self) -> Dict:
        """
        Load and cache both models with error handling
        Returns dictionary of loaded models
        """
        models = {}
        model_dir = "models"
        
        for model_key, info in _self.model_info.items():
            model_path = os.path.join(model_dir, info['filename'])
            try:
                if os.path.exists(model_path):
                    model = joblib.load(model_path)
                    models[model_key] = model
                    st.success(f"🤖\n✅ {info['name']} loaded successfully!")
                else:
                    st.warning(f"⚠️ Model file not found: {model_path}")
            except Exception as e:
                st.error(f"❌ Failed to load {info['name']}: {str(e)}")
        
        return models
    # ...inside ModelPredictor class...

    def get_model_stats(self) -> Dict:
      """Get statistics about loaded models"""
      stats = {
        'total_models': len(self.models),
        'available_models': list(self.models.keys()),
        'model_info': {}
     }
      for model_key in self.models.keys():
        stats['model_info'][model_key] = self.model_info.get(model_key, {})
      return stats

    def get_available_models(self) -> List[str]:
        """Get list of successfully loaded models"""
        return list(self.models.keys())
    
    def get_model_info(self, model_key: str) -> Dict:
        """Get information about a specific model"""
        return self.model_info.get(model_key, {})
    
    def predict_single(self, message: str, model_key: str = 'svm') -> Dict:
        """
        Predict spam/ham for a single message

        Args:
        message (str): Text message to classify
        model_key (str): Model to use ('svm' or 'lr')

        Returns:
        Dict: Prediction results with probabilities and metadata
        """
        if model_key not in self.models:
          raise ValueError(f"Model '{model_key}' not available. Available models: {self.get_available_models()}")

        if not message or len(message.strip()) < 3:
          raise ValueError("Message must be at least 3 characters long")

        try:
          model = self.models[model_key]
          start_time = time.time()

          # Get prediction and probabilities
          prediction = model.predict([message])[0]
          probabilities = model.predict_proba([message])[0]

          # Processing time
          processing_time = time.time() - start_time

          # Map prediction to label
          prediction_label = "SPAM" if prediction == 1 else "HAM"
          confidence = max(probabilities)

          # Robust class probability mapping
          classes = list(model.classes_)
          if 0 in classes and 1 in classes:
            ham_index = classes.index(0)
            spam_index = classes.index(1)
          elif 'ham' in classes and 'spam' in classes:
            ham_index = classes.index('ham')
            spam_index = classes.index('spam')
          else:
            # fallback: assign first class as ham, second as spam
            ham_index = 0
            spam_index = 1

          ham_prob = probabilities[ham_index]
          spam_prob = probabilities[spam_index]

          result = {
            'message': message,
            'prediction': prediction,
            'prediction_label': prediction_label,
            'confidence': confidence,
            'probabilities': {
                'ham': ham_prob,
                'spam': spam_prob
            },
            'model_used': model_key,
            'model_name': self.model_info[model_key]['name'],
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat()
          }

          logger.info(f"Prediction made: {prediction_label} ({confidence:.3f} confidence)")
          return result

        except Exception as e:
          logger.error(f"Prediction error: {str(e)}")
          raise Exception(f"Prediction failed: {str(e)}")
    
    def predict_batch(self, messages: List[str], model_key: str = 'svm') -> pd.DataFrame:
       """
       Predict spam/ham for multiple messages

       Args:
        messages (List[str]): List of messages to classify
        model_key (str): Model to use ('svm' or 'lr')

       Returns:
        pd.DataFrame: Results dataframe with predictions and probabilities
       """
       if model_key not in self.models:
        raise ValueError(f"Model '{model_key}' not available. Available models: {self.get_available_models()}")

       if not messages:
        raise ValueError("Messages list cannot be empty")

       # Filter out empty messages
       valid_messages = [msg for msg in messages if msg and len(msg.strip()) >= 3]

       if not valid_messages:
        raise ValueError("No valid messages found (must be at least 3 characters)")

       try:
        model = self.models[model_key]
        start_time = time.time()

        # Batch predictions
        predictions = model.predict(valid_messages)
        probabilities = model.predict_proba(valid_messages)

        processing_time = time.time() - start_time

        # Robust class probability mapping
        classes = list(model.classes_)
        if 0 in classes and 1 in classes:
            ham_index = classes.index(0)
            spam_index = classes.index(1)
        elif 'ham' in classes and 'spam' in classes:
            ham_index = classes.index('ham')
            spam_index = classes.index('spam')
        else:
            ham_index = 0
            spam_index = 1

        # Create results dataframe
        results = []
        for i, (msg, pred, probs) in enumerate(zip(valid_messages, predictions, probabilities)):
            prediction_label = "SPAM" if pred == 1 else "HAM"
            confidence = max(probs)
            ham_prob = probs[ham_index]
            spam_prob = probs[spam_index]

            results.append({
                'message_id': i + 1,
                'message': msg[:100] + "..." if len(msg) > 100 else msg,
                'full_message': msg,
                'prediction': pred,
                'prediction_label': prediction_label,
                'confidence': confidence,
                'ham_probability': ham_prob,
                'spam_probability': spam_prob,
            })

        df = pd.DataFrame(results)

        # Add metadata
        df.attrs['model_used'] = model_key
        df.attrs['model_name'] = self.model_info[model_key]['name']
        df.attrs['processing_time'] = processing_time
        df.attrs['timestamp'] = datetime.now().isoformat()
        df.attrs['total_messages'] = len(valid_messages)

        logger.info(f"Batch prediction completed: {len(valid_messages)} messages in {processing_time:.3f}s")
        return df

       except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise Exception(f"Batch prediction failed: {str(e)}")
    
    def compare_models(self, message: str) -> Dict:
        """
        Compare predictions from both models on the same message
        
        Args:
            message (str): Message to classify with both models
            
        Returns:
            Dict: Comparison results from both models
        """
        if not message or len(message.strip()) < 3:
            raise ValueError("Message must be at least 3 characters long")
        
        available_models = self.get_available_models()
        if len(available_models) < 2:
            raise ValueError(f"Need at least 2 models for comparison. Available: {len(available_models)}")
        
        comparison = {
            'message': message,
            'models': {},
            'agreement': None,
            'confidence_difference': None,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Get predictions from all available models
            for model_key in available_models:
                result = self.predict_single(message, model_key)
                comparison['models'][model_key] = {
                    'prediction_label': result['prediction_label'],
                    'confidence': result['confidence'],
                    'probabilities': result['probabilities'],
                    'model_name': result['model_name']
                }
            
            # Calculate agreement between models
            predictions = [comparison['models'][key]['prediction_label'] for key in available_models]
            comparison['agreement'] = len(set(predictions)) == 1
            
            # Calculate confidence difference (if 2 models available)
            if len(available_models) == 2:
                model_keys = list(available_models)
                conf1 = comparison['models'][model_keys[0]]['confidence']
                conf2 = comparison['models'][model_keys[1]]['confidence']
                comparison['confidence_difference'] = abs(conf1 - conf2)
            
            logger.info(f"Model comparison completed - Agreement: {comparison['agreement']}")
            return comparison
            
        except Exception as e:
            logger.error(f"Model comparison error: {str(e)}")
            raise Exception(f"Model comparison failed: {str(e)}")
    
    def get_model_stats(self) -> Dict:
        """Get statistics about loaded models"""
        stats = {
            'total_models': len(self.models),
            'available_models': list(self.models.keys()),
            'model_info': {}
        }
        
        for model_key, model in self.models.items():
            info = self.model_info[model_key]
            try:
                # You can add more model-specific stats here if needed
                stats['model_info'][model_key] = info
            except Exception as e:
                logger.error(f"Error getting stats for {model_key}: {str(e)}")
        
        return stats

# Global predictor instance
@st.cache_resource
def get_predictor():
    """Get cached predictor instance"""
    return ModelPredictor()