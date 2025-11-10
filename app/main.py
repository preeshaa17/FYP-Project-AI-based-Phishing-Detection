from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # ADD THIS LINE
from pydantic import BaseModel
import joblib
import os
from typing import List

# 🔑 CRITICAL: Import your feature building function from the local features.py file
# Ensure features.py is in the same directory as main.py
from features import build_feature_matrix 

# --- Configuration ---
app = FastAPI(title="Phishing Detection API")

# --- CORS Configuration (ADD THIS SECTION) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://localhost:8080"],  # Your Outlook add-in frontend
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Explicitly include OPTIONS
    allow_headers=["*"],
)

# --- Model Loading ---
# Assuming 'phishing_model.pkl' and 'vectorizer.pkl' are in the same directory as main.py
try:
    model = joblib.load("phishing_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    print("✅ Models loaded successfully.")
except FileNotFoundError:
    # Use absolute paths if files are in a specific 'model' subdirectory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "model")
    try:
        model = joblib.load(os.path.join(MODEL_DIR, "phishing_model.pkl"))
        vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
        print("✅ Models loaded successfully from /model directory.")
    except FileNotFoundError:
        raise RuntimeError("Model files (phishing_model.pkl or vectorizer.pkl) not found. "
                           "Ensure they are in the same directory or in a 'model/' subdirectory.")


# --- Security ---
API_KEY = "change-me"  # ⚠️ REMEMBER TO SET THIS TO A STRONG, UNIQUE KEY


# --- Request/Response Schemas ---
# Defines the expected input format from the Outlook Add-in (taskpane.js)
class EmailIn(BaseModel):
    subject: str = ""
    bodyHtml: str = ""
    bodyText: str = ""
    sender: str = ""
    urls: List[str] = []

# --- API Endpoints ---

@app.get("/health")
def health_check():
    """Simple endpoint to verify the API is running."""
    return {"status": "ok", "message": "Phishing Detection Service is operational."}

@app.post("/predict")
def predict(email: EmailIn, x_api_key: str = Header(default="")):
    """
    Receives email content and returns a phishing prediction.
    """
    # 1. Security Check
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # 2. Build the feature matrix using the function from features.py
    # This step replicates the exact feature engineering order from rfc2.py
    X_combined = build_feature_matrix(
        subject=email.subject,
        body_text=email.bodyText,
        body_html=email.bodyHtml,
        urls_list=email.urls,
        tfidf_vectorizer=vectorizer  # Pass the loaded vectorizer
    )
    
    # 3. Prediction
    # Note: [0][1] gets the probability of the positive class (Phishing=1)
    proba = model.predict_proba(X_combined)[0][1]
    
    # 4. Label based on a 0.5 threshold
    label = "phishing" if proba >= 0.5 else "legitimate"
    
    # 5. Response
    return {"prediction": label, "score": round(float(proba), 4)}