"""
utils.py - Utility functions for email spam classifier
"""

import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import joblib

def save_model(model, filename: str):
    """Save model to file"""
    joblib.dump(model, filename)
    print(f"✅ Model saved to {filename}")

def load_model(filename: str):
    """Load model from file"""
    model = joblib.load(filename)
    print(f"✅ Model loaded from {filename}")
    return model

def save_results(results: Dict, filename: str):
    """Save results to JSON file"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results saved to {filename}")

def load_results(filename: str) -> Dict:
    """Load results from JSON file"""
    with open(filename, 'r') as f:
        results = json.load(f)
    print(f"✅ Results loaded from {filename}")
    return results

def create_sample_emails() -> Dict:
    """Create sample emails for testing"""
    samples = {
        "spam": [
            "Congratulations! You've been selected to win a $1000 Walmart gift card. Click now to claim!",
            "URGENT: Your bank account has been locked. Verify your identity immediately to avoid suspension.",
            "Make $5000 weekly from home! No experience needed. Start earning today!",
            "Your computer is infected with viruses! Download our antivirus software now for protection.",
            "You've won an iPhone 15! Claim your free prize here: bit.ly/freephone123",
            "Limited time offer: Get 90% discount on all products. Shop now before it's too late!",
            "Investment opportunity: Double your money in 24 hours. Guaranteed returns!",
            "Account verification required: Your PayPal account needs immediate attention.",
            "You have unclaimed inheritance money. Contact us to claim your funds.",
            "Special bonus: Free casino credits waiting for you. Play and win big!"
        ],
        "ham": [
            "Hi team, attached is the Q3 report for your review. Please provide feedback by Friday.",
            "Meeting reminder: Project kickoff at 2 PM today in Conference Room A.",
            "Your Amazon order #12345 has shipped. Expected delivery: October 25.",
            "Password reset request for your account. Click here to reset your password.",
            "Weekly newsletter: Latest updates on our products and services.",
            "Hello John, hope you're doing well. Let's schedule a call to discuss the project.",
            "Please find attached the invoice for your recent purchase.",
            "Thank you for your application. We'll review it and get back to you soon.",
            "Company holiday schedule for December is now available on the intranet.",
            "Reminder: Team building event this Friday at 3 PM in the cafeteria."
        ]
    }
    
    # Save to file
    with open('data/sample_emails.json', 'w') as f:
        json.dump(samples, f, indent=2)
    
    return samples

def calculate_metrics(predictions: List[Dict], actual: List[int]) -> Dict:
    """Calculate performance metrics"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    pred_labels = [1 if p['is_spam'] else 0 for p in predictions]
    
    metrics = {
        'accuracy': accuracy_score(actual, pred_labels),
        'precision': precision_score(actual, pred_labels, zero_division=0),
        'recall': recall_score(actual, pred_labels, zero_division=0),
        'f1_score': f1_score(actual, pred_labels, zero_division=0),
        'total_samples': len(predictions),
        'spam_predictions': sum(pred_labels),
        'actual_spam': sum(actual)
    }
    
    return metrics

def print_predictions_table(predictions: List[Dict]):
    """Print predictions in a nice table format"""
    import pandas as pd
    
    df = pd.DataFrame(predictions)
    print("\n📊 PREDICTIONS TABLE:")
    print("-" * 80)
    print(df[['prediction', 'spam_probability', 'confidence']].to_string())
    
    # Summary
    spam_count = sum(1 for p in predictions if p['is_spam'])
    print(f"\n📈 SUMMARY: {spam_count}/{len(predictions)} emails classified as SPAM")
    
    return df

def get_model_performance() -> Dict:
    """Get model performance from saved report"""
    try:
        with open('models/model_performance_report.json', 'r') as f:
            report = json.load(f)
        
        print("\n📈 MODEL PERFORMANCE REPORT:")
        print("=" * 50)
        print(f"Accuracy: {report.get('accuracy', 0)*100:.2f}%")
        print(f"Precision: {report.get('precision', 0)*100:.2f}%")
        print(f"Recall: {report.get('recall', 0)*100:.2f}%")
        print(f"F1-Score: {report.get('f1_score', 0)*100:.2f}%")
        
        if 'confusion_matrix' in report:
            print(f"\nConfusion Matrix:")
            cm = report['confusion_matrix']
            print(f"  True Negatives: {cm[0][0]}")
            print(f"  False Positives: {cm[0][1]}")
            print(f"  False Negatives: {cm[1][0]}")
            print(f"  True Positives: {cm[1][1]}")
        
        return report
    except:
        print("⚠️ Performance report not found")
        return {}


if __name__ == "__main__":
    # Create sample data
    samples = create_sample_emails()
    print(f"✅ Created {len(samples['spam'])} spam and {len(samples['ham'])} ham samples")
    
    # Show model performance
    get_model_performance()