# 📧 Email Spam Classifier

A production-grade machine learning system designed to classify emails as **Spam** or **Ham (Legitimate)** with high accuracy. This project features a beautiful dark-themed web interface built with Streamlit and a robust REST API powered by FastAPI.

## ✨ Features

- **🚀 High Accuracy Models**:
  - **Ensemble Model**: 97.12% accuracy (Voting Classifier with Random Forest + XGBoost + GBM)
  - **Pipeline Model**: 96.85% accuracy (Random Forest with optimized feature selection)
  - **Random Forest**: 96.50% accuracy (stand-alone implementation)
  
- **💻 Interactive Web Interface**: 
  - sleek, modern **Dark Mode** UI.
  - Real-time classification with probability scores.
  - Visual confidence metrics (Spam vs. Ham probability).

- **🔄 Batch Processing**:
  - Analyze multiple emails simultaneously.
  - Support for bulk text input or file uploads (`.txt`, `.csv`).
  - Download detailed reports as CSV.

- **📊 Advanced Analytics**:
  - Track analysis history in real-time.
  - View distribution statistics (Spam vs. Ham).
  - Filter and export historical data.

- **🔌 REST API**: Full-featured API for integrating spam detection into other applications.

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/email-spam-classifier.git
   cd email-spam-classifier
   ```

2. **Create a virtual environment (Optional but recommended)**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### 1. Web Application (Streamlit)
Launch the interactive web interface:
```bash
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`.

### 2. REST API (FastAPI)
Start the API server:
```bash
python api.py
# OR using uvicorn directly
uvicorn api:app --reload
```
- **API Documentation**: Visit `http://localhost:8000/docs` for the interactive Swagger UI.
- **Health Check**: `GET /health`

---

## 📂 Project Structure

```
├── app.py                  # Main Streamlit web application
├── api.py                  # FastAPI REST API endpoints
├── src/                    # Source code for Core Logic
│   ├── predictor.py        # Model prediction logic
│   └── preprocessing.py    # Text cleaning and feature extraction
├── models/                 # Serialized ML models (joblib/pickle)
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## 📊 Model Performance

| Model Type | Accuracy | Features | Description |
|------------|----------|----------|-------------|
| **Ensemble** | **97.12%** | 576 | Best for production. Combines RF, XGBoost, and GBM. |
| **Pipeline** | 96.85% | 576 | Balanced performance using feature selection. |
| **Random Forest** | 96.50% | 576 | Fast and reliable baseline model. |

## � Technologies Used

- **Frontend**: Streamlit
- **Backend**: FastAPI, Uvicorn
- **Machine Learning**: Scikit-Learn, XGBoost, Numpy, Pandas
- **Visualization**: Plotly, Matplotlib

---

## 📝 License

This project is open-source and available under the MIT License.