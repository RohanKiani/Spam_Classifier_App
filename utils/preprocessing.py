"""
Text Preprocessing Module for Spam Detection System
Professional text cleaning, normalization, and feature extraction utilities
"""

import re
import string
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import logging
from datetime import datetime

# =============================================================================
# NLTK DATA DOWNLOAD WITH CACHING
# =============================================================================

@st.cache_resource
def download_nltk_data():
    """Download required NLTK data with caching and error handling"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        return True
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            return True
        except Exception as e:
            logging.error(f"Failed to download NLTK data: {e}")
            return False

# =============================================================================
# PREPROCESSING CLASS
# =============================================================================

class TextPreprocessor:
    """
    Professional text preprocessing class for spam detection
    Handles cleaning, normalization, and feature extraction
    """
    
    def __init__(self, 
                 remove_stopwords: bool = True,
                 use_stemming: bool = False,
                 use_lemmatization: bool = True,
                 min_length: int = 3,
                 max_length: int = 10000):
        """
        Initialize text preprocessor with configurable options
        
        Args:
            remove_stopwords: Whether to remove English stopwords
            use_stemming: Whether to apply stemming (not recommended with lemmatization)
            use_lemmatization: Whether to apply lemmatization
            min_length: Minimum message length to process
            max_length: Maximum message length to process
        """
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.min_length = min_length
        self.max_length = max_length
        
        # Initialize NLTK components
        self.nltk_ready = download_nltk_data()
        if self.nltk_ready:
            self.stop_words = set(stopwords.words('english'))
            self.stemmer = PorterStemmer()
            self.lemmatizer = WordNetLemmatizer()
        else:
            st.warning("⚠️ NLTK data not available. Using basic preprocessing only.")
            self.stop_words = set()
            self.stemmer = None
            self.lemmatizer = None
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text with comprehensive preprocessing
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned and normalized text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove extra whitespace and line breaks
        text = re.sub(r'\s+', ' ', text)
        
        # Handle URLs - replace with generic token
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 
                     ' URL_TOKEN ', text)
        
        # Handle email addresses - replace with generic token
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
                     ' EMAIL_TOKEN ', text)
        
        # Handle phone numbers - replace with generic token
        text = re.sub(r'(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}', 
                     ' PHONE_TOKEN ', text)
        
        # Handle money amounts - replace with generic token
        text = re.sub(r'\$\d+(?:\.\d{2})?|\d+\s*(?:dollars?|usd|pounds?|euros?)', 
                     ' MONEY_TOKEN ', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Handle contractions
        contractions = {
            "don't": "do not", "won't": "will not", "can't": "cannot",
            "n't": " not", "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am", "'s": " is"
        }
        for contraction, expansion in contractions.items():
            text = re.sub(contraction, expansion, text)
        
        # Remove punctuation but keep sentence structure
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_process(self, text: str) -> List[str]:
        """
        Tokenize text and apply advanced NLP processing
        
        Args:
            text: Cleaned text string
            
        Returns:
            List of processed tokens
        """
        if not text or not self.nltk_ready:
            return text.split()
        
        try:
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords if enabled
            if self.remove_stopwords:
                tokens = [token for token in tokens if token.lower() not in self.stop_words]
            
            # Filter tokens (remove very short words and numbers only)
            tokens = [token for token in tokens if len(token) >= 2 and not token.isdigit()]
            
            # Apply stemming or lemmatization
            if self.use_lemmatization and self.lemmatizer:
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            elif self.use_stemming and self.stemmer:
                tokens = [self.stemmer.stem(token) for token in tokens]
            
            return tokens
            
        except Exception as e:
            logging.error(f"Error in tokenization: {e}")
            return text.split()
    
    def preprocess_message(self, message: str) -> str:
        """
        Complete preprocessing pipeline for a single message
        
        Args:
            message: Raw message text
            
        Returns:
            Fully preprocessed message ready for model input
        """
        # Validate input
        if not message or not isinstance(message, str):
            return ""
        
        # Check length constraints
        if len(message) < self.min_length:
            return ""
        
        if len(message) > self.max_length:
            message = message[:self.max_length]
        
        # Clean text
        cleaned = self.clean_text(message)
        
        # Tokenize and process
        tokens = self.tokenize_and_process(cleaned)
        
        # Rejoin tokens
        processed = " ".join(tokens)
        
        return processed.strip()
    
    def preprocess_batch(self, messages: List[str], show_progress: bool = True) -> List[str]:
        """
        Preprocess a batch of messages with optional progress tracking
        
        Args:
            messages: List of raw message strings
            show_progress: Whether to show Streamlit progress bar
            
        Returns:
            List of preprocessed messages
        """
        if not messages:
            return []
        
        processed_messages = []
        
        if show_progress:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        for i, message in enumerate(messages):
            if show_progress:
                progress = (i + 1) / len(messages)
                progress_bar.progress(progress)
                status_text.text(f"Processing message {i+1}/{len(messages)}")
            
            processed = self.preprocess_message(message)
            processed_messages.append(processed)
        
        if show_progress:
            progress_bar.empty()
            status_text.empty()
            st.success(f"✅ Successfully processed {len(messages)} messages!")
        
        return processed_messages

# =============================================================================
# FEATURE EXTRACTION UTILITIES
# =============================================================================

def extract_message_features(message: str) -> Dict[str, Union[int, float]]:
    """
    Extract statistical features from message text
    
    Args:
        message: Input message text
        
    Returns:
        Dictionary of extracted features
    """
    if not message:
        return {
            'char_count': 0, 'word_count': 0, 'sentence_count': 0,
            'avg_word_length': 0, 'caps_ratio': 0, 'digit_ratio': 0,
            'special_char_ratio': 0, 'url_count': 0, 'email_count': 0
        }
    
    features = {}
    
    # Basic counts
    features['char_count'] = len(message)
    features['word_count'] = len(message.split())
    features['sentence_count'] = len([s for s in re.split(r'[.!?]+', message) if s.strip()])
    
    # Word statistics
    words = message.split()
    features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
    
    # Character ratios
    features['caps_ratio'] = sum(1 for c in message if c.isupper()) / len(message) if message else 0
    features['digit_ratio'] = sum(1 for c in message if c.isdigit()) / len(message) if message else 0
    features['special_char_ratio'] = sum(1 for c in message if c in string.punctuation) / len(message) if message else 0
    
    # Special patterns
    features['url_count'] = len(re.findall(r'http[s]?://[^\s]+', message))
    features['email_count'] = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', message))
    
    return features

def batch_feature_extraction(messages: List[str]) -> pd.DataFrame:
    """
    Extract features from a batch of messages
    
    Args:
        messages: List of message strings
        
    Returns:
        DataFrame with extracted features
    """
    features_list = []
    for message in messages:
        features = extract_message_features(message)
        features_list.append(features)
    
    return pd.DataFrame(features_list)

# =============================================================================
# TEXT ANALYSIS UTILITIES
# =============================================================================

def analyze_text_quality(text: str) -> Dict[str, Union[str, float, int]]:
    """
    Analyze text quality and provide insights
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary with quality metrics and recommendations
    """
    analysis = {
        'original_length': len(text) if text else 0,
        'word_count': len(text.split()) if text else 0,
        'quality_score': 0.0,
        'issues': [],
        'recommendations': []
    }
    
    if not text:
        analysis['quality_score'] = 0.0
        analysis['issues'].append("Empty text")
        return analysis
    
    # Calculate quality score
    score = 100.0
    
    # Length checks
    if len(text) < 10:
        score -= 30
        analysis['issues'].append("Text too short")
        analysis['recommendations'].append("Provide more detailed text for better analysis")
    
    if len(text) > 5000:
        score -= 10
        analysis['issues'].append("Text very long")
    
    # Word count checks
    word_count = len(text.split())
    if word_count < 3:
        score -= 25
        analysis['issues'].append("Too few words")
    
    # Special character ratio
    special_ratio = sum(1 for c in text if c in string.punctuation) / len(text)
    if special_ratio > 0.3:
        score -= 15
        analysis['issues'].append("Too many special characters")
    
    # All caps check
    caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
    if caps_ratio > 0.5:
        score -= 20
        analysis['issues'].append("Excessive capitalization")
        analysis['recommendations'].append("Reduce capitalization for better readability")
    
    analysis['quality_score'] = max(0, score)
    
    if not analysis['issues']:
        analysis['recommendations'].append("Text quality is good for analysis")
    
    return analysis

# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def validate_message_input(message: str) -> Tuple[bool, str]:
    """
    Validate message input for processing
    
    Args:
        message: Input message to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not message:
        return False, "Message cannot be empty"
    
    if not isinstance(message, str):
        return False, "Message must be text"
    
    if len(message.strip()) < 3:
        return False, "Message too short (minimum 3 characters)"
    
    if len(message) > 10000:
        return False, "Message too long (maximum 10,000 characters)"
    
    # Check for meaningful content
    if message.strip().isdigit():
        return False, "Message cannot contain only numbers"
    
    return True, "Valid message"

def validate_batch_input(messages: List[str]) -> Tuple[bool, str, List[int]]:
    """
    Validate batch message input
    
    Args:
        messages: List of messages to validate
        
    Returns:
        Tuple of (is_valid, error_message, invalid_indices)
    """
    if not messages:
        return False, "No messages provided", []
    
    if len(messages) > 1000:
        return False, "Too many messages (maximum 1000)", []
    
    invalid_indices = []
    for i, message in enumerate(messages):
        is_valid, _ = validate_message_input(message)
        if not is_valid:
            invalid_indices.append(i)
    
    if len(invalid_indices) > len(messages) * 0.5:
        return False, f"Too many invalid messages ({len(invalid_indices)} out of {len(messages)})", invalid_indices
    
    return True, f"Valid batch with {len(messages) - len(invalid_indices)} processable messages", invalid_indices

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

@st.cache_data
def get_preprocessing_stats() -> Dict[str, any]:
    """Get preprocessing statistics and configuration"""
    return {
        'nltk_available': download_nltk_data(),
        'supported_features': [
            'Text cleaning and normalization',
            'URL, email, and phone number tokenization',
            'Stopword removal',
            'Lemmatization and stemming',
            'Feature extraction',
            'Batch processing'
        ],
        'preprocessing_steps': [
            'Convert to lowercase',
            'Remove extra whitespace',
            'Handle special tokens (URLs, emails, etc.)',
            'Remove HTML tags',
            'Expand contractions',
            'Remove punctuation',
            'Tokenization',
            'Stopword removal (optional)',
            'Lemmatization (optional)'
        ]
    }

def create_default_preprocessor() -> TextPreprocessor:
    """Create a default preprocessor with recommended settings"""
    return TextPreprocessor(
        remove_stopwords=True,
        use_stemming=False,
        use_lemmatization=True,
        min_length=3,
        max_length=10000
    )

# =============================================================================
# TESTING AND EXAMPLES
# =============================================================================

def get_preprocessing_examples() -> Dict[str, str]:
    """Get example messages for testing preprocessing"""
    return {
        "spam_example": "URGENT!!! Click here http://fake-site.com to WIN $1000000!!! Don't miss this AMAZING opportunity! Call 1-800-SCAM-NOW!!!",
        "ham_example": "Hey, can we meet tomorrow at 3 PM to discuss the project? Let me know if that works for you. Thanks!",
        "mixed_example": "Your account has been LOCKED! Please visit our website immediately to verify your information at secure-bank-login.com or call us at 555-0123."
    }

if __name__ == "__main__":
    # Example usage
    preprocessor = create_default_preprocessor()
    examples = get_preprocessing_examples()
    
    print("=== Text Preprocessing Demo ===")
    for name, text in examples.items():
        print(f"\n{name.upper()}:")
        print(f"Original: {text}")
        processed = preprocessor.preprocess_message(text)
        print(f"Processed: {processed}")
        features = extract_message_features(text)
        print(f"Features: {features}")