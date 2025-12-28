"""
preprocessing.py - Text preprocessing utilities
"""

import re
import string
from typing import List, Dict
import pandas as pd
import numpy as np

class TextPreprocessor:
    """Preprocess email text for spam classification"""
    
    def __init__(self):
        # Common spam indicators
        self.spam_indicators = {
            'free': 3, 'win': 3, 'prize': 3, 'money': 3, 'cash': 3,
            'click': 2, 'urgent': 2, 'guaranteed': 2, 'risk': 2,
            'winner': 3, 'congratulations': 3, 'deal': 2, 'credit': 2,
            'loan': 2, 'mortgage': 2, 'viagra': 4, 'casino': 4,
            'lottery': 3, 'inheritance': 3, 'unclaimed': 3,
            'discount': 2, 'offer': 2, 'limited': 2, 'special': 1,
            'bonus': 2, 'reward': 2, 'selected': 1, 'exclusive': 1
        }
        
        # Common ham indicators
        self.ham_indicators = {
            'meeting': 1, 'report': 1, 'attached': 1, 'please': 1,
            'review': 1, 'team': 1, 'project': 1, 'agenda': 1,
            'minutes': 1, 'follow': 1, 'discussion': 1, 'update': 1,
            'schedule': 1, 'confirm': 1, 'attachment': 1, 'regards': 1,
            'thanks': 1, 'best': 1, 'kind': 1, 'sincerely': 1,
            'hello': 1, 'hi': 1, 'dear': 1, 'colleague': 1,
            'work': 1, 'office': 1, 'business': 1, 'professional': 1
        }
    
    def clean_text(self, text: str) -> str:
        """Clean email text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove email headers
        text = re.sub(r'from:.+?\n', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'to:.+?\n', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'subject:.+?\n', ' ', text, flags=re.IGNORECASE)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', ' URL ', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', ' EMAIL ', text)
        
        # Remove special characters (keep basic punctuation)
        text = re.sub(r'[^a-zA-Z0-9\s.!?]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_features(self, text: str) -> Dict:
        """Extract features from text"""
        cleaned = self.clean_text(text)
        words = cleaned.split()
        
        # Calculate spam score
        spam_score = 0
        for word in words:
            if word in self.spam_indicators:
                spam_score += self.spam_indicators[word]
        
        # Calculate ham score
        ham_score = 0
        for word in words:
            if word in self.ham_indicators:
                ham_score += self.ham_indicators[word]
        
        # Text statistics
        num_words = len(words)
        num_chars = len(text)
        num_sentences = len(re.split(r'[.!?]+', text))
        
        # Spam indicators
        num_exclamation = text.count('!')
        num_dollar = text.count('$')
        num_uppercase = sum(1 for c in text if c.isupper())
        uppercase_ratio = num_uppercase / len(text) if len(text) > 0 else 0
        
        # Word frequency
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Most common words
        top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10])
        
        return {
            'text_cleaned': cleaned,
            'num_words': num_words,
            'num_chars': num_chars,
            'num_sentences': num_sentences,
            'spam_score': spam_score,
            'ham_score': ham_score,
            'spam_ham_ratio': spam_score / (ham_score + 1),
            'num_exclamation': num_exclamation,
            'num_dollar': num_dollar,
            'uppercase_ratio': uppercase_ratio,
            'top_words': top_words,
            'has_spam_keywords': spam_score > 5,
            'has_urgent': 'urgent' in cleaned
        }
    
    def create_feature_vector(self, text: str) -> pd.DataFrame:
        """Create feature vector for model prediction"""
        features = self.extract_features(text)
        
        # Create a simple feature vector
        # This is a simplified version - real version would use the full vocabulary
        feature_dict = {
            'length': features['num_words'],
            'spam_score': features['spam_score'],
            'exclamation_count': features['num_exclamation'],
            'dollar_count': features['num_dollar'],
            'uppercase_ratio': features['uppercase_ratio'],
            'spam_ham_ratio': features['spam_ham_ratio']
        }
        
        # Add keyword presence features
        for keyword in ['free', 'win', 'money', 'click', 'urgent']:
            feature_dict[f'has_{keyword}'] = 1 if keyword in text.lower() else 0
        
        return pd.DataFrame([feature_dict])


def test_preprocessor():
    """Test the preprocessor"""
    preprocessor = TextPreprocessor()
    
    test_text = "WINNER!! You've won a FREE $1000 gift card! Click HERE to claim now!"
    
    features = preprocessor.extract_features(test_text)
    
    print("Text Features:")
    for key, value in features.items():
        if key != 'text_cleaned' and key != 'top_words':
            print(f"  {key}: {value}")
    
    print(f"\nTop words: {features['top_words']}")
    
    return features


if __name__ == "__main__":
    test_preprocessor()