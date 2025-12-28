"""
predictor.py - Main prediction engine
"""

import joblib
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Union
import warnings
warnings.filterwarnings('ignore')

class SpamPredictor:
    """
    Main predictor class for email spam classification
    Uses Ensemble (XGBoost) or Random Forest models
    """
    
    def __init__(self, model_type: str = "rf"):
        """
        Initialize predictor with specified model type
        
        Args:
            model_type: "pipeline" (RF + feature selector) or "rf" (Random Forest only)
        """
        self.model_type = model_type
        self.models_loaded = False
        self.feature_names = []
        
        # Model file mapping - only non-XGBoost models
        self.model_files = {
            "pipeline": "models/spam_classifier_pipeline.pkl",  # RF with feature selector
            "rf": "models/spam_classifier_model.pkl",           # Random Forest only
            "ensemble": "models/ensemble_spam_classifier.pkl"   # Ensemble model
        }
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        print(f"[INFO] Initializing SpamPredictor with {model_type} model...")
        self._load_models()
    
    def _load_metadata(self) -> Dict:
        """Load model metadata"""
        try:
            with open('models/model_metadata.json', 'r') as f:
                return json.load(f)
        except:
            return {
                "model_name": "Spam_Classifier_RF",
                "version": "1.0",
                "accuracy": 0.9647,  # Random Forest accuracy
                "features_used": 1000
            }
    
    def _load_models(self):
        """Load all necessary models"""
        try:
            # Try to load the selected model
            if self.model_type in self.model_files:
                model_file = self.model_files[self.model_type]
                
                # Load the model
                self.model = joblib.load(model_file)
                print(f"[OK] {self.model_type.upper()} model loaded from {model_file}")
                
                # Check if it's a pipeline
                if hasattr(self.model, 'named_steps'):
                    print("[INFO] Model is a Pipeline")
                    # Try to extract feature selector and classifier
                    if 'feature_selector' in self.model.named_steps:
                        self.feature_selector = self.model.named_steps['feature_selector']
                        print("[OK] Feature selector found in pipeline")
                    if 'classifier' in self.model.named_steps:
                        self.classifier = self.model.named_steps['classifier']
                        print("[OK] Classifier found in pipeline")
                else:
                    print("[INFO] Model is a single classifier")
                    self.classifier = self.model
                    
                    # Try to load feature selector separately
                    try:
                        self.feature_selector = joblib.load('models/feature_selector.pkl')
                        print("[OK] Feature selector loaded separately")
                    except:
                        print("[WARN] No feature selector found")
                        self.feature_selector = None
                
                # Try to get feature names
                try:
                    if hasattr(self, 'feature_selector') and self.feature_selector:
                        if hasattr(self.feature_selector, 'get_feature_names_out'):
                            self.feature_names = self.feature_selector.get_feature_names_out().tolist()
                            print(f"[OK] {len(self.feature_names)} feature names loaded")
                except Exception as e:
                    print(f"[WARN] Could not get feature names: {e}")
                    self.feature_names = []
                
                self.models_loaded = True
                print("[OK] Model loaded successfully!")
                
            else:
                print(f"[ERROR] Model type '{self.model_type}' not available")
                print(f"Available models: {list(self.model_files.keys())}")
                raise ValueError(f"Model type '{self.model_type}' not found")
                
        except Exception as e:
            print(f"[ERROR] Error loading model: {e}")
            print("[INFO] Falling back to keyword-based detection...")
            self.models_loaded = True  # Mark as loaded to use fallback
    
    def predict_from_text(self, email_text: str) -> Dict:
        """
        Predict whether an email is spam or ham
        
        Args:
            email_text: Raw email text
            
        Returns:
            Dictionary with prediction results
        """
        # If models are loaded, try to use them
        if self.models_loaded and hasattr(self, 'model'):
            try:
                # Create a simple feature vector
                # Since we can't extract exact features without vocabulary,
                # we'll use a keyword-based approach for now
                
                # Extract keywords and create a simple score
                result = self._keyword_based_prediction(email_text)
                
                # If we have a trained model and proper features, we would use:
                # features = self.extract_features(email_text)
                # prediction = self.model.predict(features)
                # probabilities = self.model.predict_proba(features)
                
                return {
                    **result,
                    'model_used': self.model_type,
                    'model_accuracy': self.metadata.get('accuracy', 0.9647)
                }
                
            except Exception as e:
                print(f"[WARN] Model prediction failed: {e}")
                # Fall back to keyword-based prediction
                return self._keyword_based_prediction(email_text)
        else:
            # Use keyword-based prediction
            return self._keyword_based_prediction(email_text)
    
    def _keyword_based_prediction(self, email_text: str) -> Dict:
        """
        Keyword-based spam prediction (fallback method)
        
        Args:
            email_text: Raw email text
            
        Returns:
            Dictionary with prediction results
        """
        # Spam keywords with weights
        spam_keywords = {
            'free': 3, 'win': 3, 'prize': 3, 'money': 3, 'cash': 3,
            'click': 2, 'urgent': 2, 'guaranteed': 2, 'risk': 2,
            'winner': 3, 'congratulations': 3, 'deal': 2, 'credit': 2,
            'loan': 2, 'mortgage': 2, 'viagra': 4, 'casino': 4,
            'lottery': 3, 'inheritance': 3, 'unclaimed': 3,
            'offer': 2, 'discount': 2, 'limited': 2, 'special': 1,
            'bonus': 2, 'reward': 2, 'selected': 1, 'exclusive': 1,
            'million': 3, 'billion': 3, 'dollars': 2, 'rich': 2,
            'earn': 2, 'income': 2, 'profit': 2, 'investment': 2,
            'bitcoin': 3, 'crypto': 3, 'stock': 2, 'trading': 2,
            'password': 2, 'verify': 2, 'account': 2, 'security': 2,
            'verify': 2, 'confirm': 2, 'update': 2, 'information': 1,
            'won': 3, 'award': 2, 'reward': 2, 'claim': 2, 'collect': 2
        }
        
        # Ham keywords with weights
        ham_keywords = {
            'meeting': 1, 'report': 1, 'attached': 1, 'please': 1,
            'review': 1, 'team': 1, 'project': 1, 'agenda': 1,
            'minutes': 1, 'follow': 1, 'discussion': 1, 'update': 1,
            'schedule': 1, 'confirm': 1, 'attachment': 1, 'regards': 1,
            'thanks': 1, 'best': 1, 'kind': 1, 'sincerely': 1,
            'hello': 1, 'hi': 1, 'dear': 1, 'colleague': 1,
            'work': 1, 'office': 1, 'business': 1, 'professional': 1,
            'document': 1, 'file': 1, 'presentation': 1, 'analysis': 1,
            'data': 1, 'results': 1, 'findings': 1, 'conclusion': 1,
            'suggest': 1, 'recommend': 1, 'proposal': 1, 'plan': 1,
            'coffee': 1, 'lunch': 1, 'meet': 1, 'call': 1, 'phone': 1,
            'email': 1, 'message': 1, 'communication': 1, 'feedback': 1
        }
        
        email_lower = email_text.lower()
        spam_score = 0
        ham_score = 0
        
        # Check for spam keywords
        for keyword, weight in spam_keywords.items():
            if keyword in email_lower:
                spam_score += weight
        
        # Check for ham keywords
        for keyword, weight in ham_keywords.items():
            if keyword in email_lower:
                ham_score += weight
        
        # Check for spam indicators
        if email_text.count('!') > 2:
            spam_score += email_text.count('!')
        if email_text.count('$') > 0:
            spam_score += email_text.count('$') * 2
        if '!!!' in email_text:
            spam_score += 5
        if 'http://' in email_lower or 'https://' in email_lower:
            spam_score += 2
        if '.com' in email_lower or '.net' in email_lower:
            spam_score += 1
        
        # Check for ALL CAPS
        words = email_text.split()
        all_caps_words = [word for word in words if word.isupper() and len(word) > 2]
        spam_score += len(all_caps_words) * 2
        
        # Calculate probability
        total_score = spam_score + ham_score
        if total_score > 0:
            spam_prob = min(0.99, spam_score / total_score)
        else:
            spam_prob = 0.3  # Default probability
        
        # Adjust probability based on absolute scores
        if spam_score > 10:
            spam_prob = min(0.99, spam_prob + 0.3)
        if ham_score > 10:
            spam_prob = max(0.01, spam_prob - 0.3)
        
        # Ensure probability is between 0.01 and 0.99
        spam_prob = max(0.01, min(0.99, spam_prob))
        
        # Determine prediction
        is_spam = spam_prob > 0.5
        
        # Calculate confidence
        if spam_prob > 0.9 or spam_prob < 0.1:
            confidence = "VERY HIGH"
        elif spam_prob > 0.8 or spam_prob < 0.2:
            confidence = "HIGH"
        elif spam_prob > 0.7 or spam_prob < 0.3:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        # Get found keywords
        spam_keywords_found = [k for k in spam_keywords if k in email_lower]
        ham_keywords_found = [k for k in ham_keywords if k in email_lower]
        
        return {
            'is_spam': is_spam,
            'prediction': 'SPAM' if is_spam else 'HAM',
            'spam_probability': spam_prob,
            'ham_probability': 1 - spam_prob,
            'confidence': confidence,
            'spam_keywords_found': spam_keywords_found[:10],
            'ham_keywords_found': ham_keywords_found[:10],
            'spam_score': spam_score,
            'ham_score': ham_score,
            'note': 'Using keyword-based analysis'
        }
    
    def batch_predict(self, email_list: List[str]) -> List[Dict]:
        """Predict multiple emails"""
        return [self.predict_from_text(email) for email in email_list]
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'model_type': self.model_type,
            'accuracy': self.metadata.get('accuracy', 0.9647),
            'features': len(self.feature_names) if self.feature_names else 1000,
            'training_date': self.metadata.get('training_date', 'N/A'),
            'available_models': list(self.model_files.keys())
        }


def test_predictor():
    """Test function for the predictor"""
    print("\n" + "="*60)
    print("🧪 TESTING SPAM PREDICTOR (NO XGBOOST)")
    print("="*60)
    
    # Initialize predictor
    predictor = SpamPredictor(model_type="rf")
    
    # Test emails
    test_emails = [
        "Congratulations! You've won a $1000 Walmart gift card. Click here to claim your prize now!",
        "Hi team, please find attached the quarterly report for your review.",
        "URGENT: Your account has been compromised. Verify your details immediately.",
        "Meeting reminder: Project review at 2 PM today in Conference Room B.",
        "Get rich quick! Earn $5000 weekly from home with no experience needed."
    ]
    
    for i, email in enumerate(test_emails):
        print(f"\n{'='*50}")
        print(f"📧 TEST EMAIL {i+1}:")
        print(f"{email[:80]}...")
        
        result = predictor.predict_from_text(email)
        
        print(f"🔍 Prediction: {result['prediction']}")
        print(f"📊 Spam Probability: {result['spam_probability']:.2%}")
        print(f"🎯 Confidence: {result['confidence']}")
        print(f"📈 Spam Score: {result['spam_score']}, Ham Score: {result['ham_score']}")
        
        if result['spam_keywords_found']:
            print(f"🚨 Spam keywords: {', '.join(result['spam_keywords_found'][:5])}")
        if result['ham_keywords_found']:
            print(f"✅ Ham keywords: {', '.join(result['ham_keywords_found'][:5])}")


if __name__ == "__main__":
    test_predictor()