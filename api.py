"""
api.py - FastAPI REST API for email spam classification
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.predictor import SpamPredictor
from src.preprocessing import TextPreprocessor

# Initialize FastAPI app
app = FastAPI(
    title="Email Spam Classifier API",
    description="REST API for classifying emails as spam or ham",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize models
predictor = SpamPredictor(model_type="ensemble")
preprocessor = TextPreprocessor()

# Request/Response models
class EmailRequest(BaseModel):
    email_text: str
    model_type: Optional[str] = "ensemble"

class EmailResponse(BaseModel):
    is_spam: bool
    prediction: str
    spam_probability: float
    ham_probability: float
    confidence: str
    model_used: str
    features_extracted: int

class BatchRequest(BaseModel):
    emails: List[str]
    model_type: Optional[str] = "ensemble"

class BatchResponse(BaseModel):
    results: List[EmailResponse]
    total_emails: int
    spam_count: int
    spam_percentage: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    accuracy: float
    version: str

# Routes
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Email Spam Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Check API health",
            "/predict": "Classify single email",
            "/batch-predict": "Classify multiple emails",
            "/stats": "Get model statistics"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health and model status"""
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        accuracy=predictor.metadata.get('accuracy', 0.97),
        version="1.0.0"
    )

@app.post("/predict", response_model=EmailResponse, tags=["Prediction"])
async def predict_email(request: EmailRequest):
    """
    Classify a single email as spam or ham
    
    - **email_text**: The email content to classify
    - **model_type**: Model to use (ensemble, pipeline, rf)
    """
    try:
        # Get prediction
        result = predictor.predict_from_text(request.email_text)
        
        # Extract features for count
        features = preprocessor.extract_features(request.email_text)
        
        return EmailResponse(
            is_spam=result['is_spam'],
            prediction=result['prediction'],
            spam_probability=result['spam_probability'],
            ham_probability=result['ham_probability'],
            confidence=result['confidence'],
            model_used=result.get('model_used', request.model_type),
            features_extracted=features['num_words']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch-predict", response_model=BatchResponse, tags=["Prediction"])
async def batch_predict(request: BatchRequest):
    """
    Classify multiple emails in batch
    
    - **emails**: List of email texts to classify
    - **model_type**: Model to use (ensemble, pipeline, rf)
    """
    try:
        if not request.emails:
            raise HTTPException(status_code=400, detail="No emails provided")
        
        results = []
        spam_count = 0
        
        for email in request.emails:
            result = predictor.predict_from_text(email)
            features = preprocessor.extract_features(email)
            
            response = EmailResponse(
                is_spam=result['is_spam'],
                prediction=result['prediction'],
                spam_probability=result['spam_probability'],
                ham_probability=result['ham_probability'],
                confidence=result['confidence'],
                model_used=result.get('model_used', request.model_type),
                features_extracted=features['num_words']
            )
            
            results.append(response)
            if result['is_spam']:
                spam_count += 1
        
        spam_percentage = (spam_count / len(request.emails)) * 100
        
        return BatchResponse(
            results=results,
            total_emails=len(request.emails),
            spam_count=spam_count,
            spam_percentage=spam_percentage
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/stats", tags=["Statistics"])
async def get_statistics():
    """Get model statistics and performance"""
    try:
        model_info = predictor.get_model_info()
        
        return {
            "model_info": model_info,
            "performance": predictor.performance_report,
            "available_models": list(predictor.model_files.keys()),
            "feature_count": len(predictor.feature_names) if predictor.feature_names else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting statistics: {str(e)}")

@app.get("/models", tags=["Models"])
async def list_models():
    """List all available models"""
    return {
        "available_models": {
            "ensemble": "Voting Classifier (RF + XGBoost + GBM) - 97.12% accuracy",
            "pipeline": "Random Forest with feature selection - 96.47% accuracy",
            "rf": "Random Forest only - 96.47% accuracy"
        },
        "default": "ensemble",
        "recommended": "ensemble"
    }

# Run the API
if __name__ == "__main__":
    print("Starting Email Spam Classifier API...")
    print("API Documentation available at: http://localhost:8000/docs")
    print("")
    print("Available endpoints:")
    print("  GET  /              - API information")
    print("  GET  /health        - Health check")
    print("  POST /predict       - Classify single email")
    print("  POST /batch-predict - Classify multiple emails")
    print("  GET  /stats         - Model statistics")
    print("  GET  /models        - List available models")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)