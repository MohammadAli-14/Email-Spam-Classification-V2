
import joblib
import sys
import os

# Create a dummy class to mock the one expected by the pickle if needed, 
# though usually joblib loads basic sklearn objects fine if versions match.

def test_load(name, path):
    print(f"\nTesting {name} at {path}...")
    if not os.path.exists(path):
        print("❌ File not found")
        return False
    
    try:
        model = joblib.load(path)
        print(f"✅ Loaded {name} successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to load {name}: {e}")
        return False

if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    try:
        import xgboost
        print(f"XGBoost version: {xgboost.__version__}")
    except ImportError:
        print("XGBoost not installed")
        
    try:
        import sklearn
        print(f"Scikit-learn version: {sklearn.__version__}")
    except ImportError:
        print("Scikit-learn not installed")

    models = {
        "rf": "models/spam_classifier_model.pkl",
        "pipeline": "models/spam_classifier_pipeline.pkl",
        "ensemble": "models/ensemble_spam_classifier.pkl"
    }

    results = {}
    for name, path in models.items():
        results[name] = test_load(name, path)
    
    print("\nSummary:")
    print(results)
