from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import joblib
from bs4 import BeautifulSoup

app = FastAPI(title="Phishing Detection API")

# Load your trained model and vectorizer
model = joblib.load("rf_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

API_KEY = "change-me"  # Optional: Add basic security if you want

# Define expected request body
class EmailIn(BaseModel):
    subject: str = ""
    bodyHtml: str = ""
    bodyText: str = ""
    sender: str = ""
    urls: list[str] = []

def extract_text(html, plain):
    if plain:
        return plain
    if html:
        try:
            return BeautifulSoup(html, "html.parser").get_text(" ")
        except Exception:
            return html
    return ""

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(email: EmailIn, x_api_key: str = Header(default="")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    text = (email.subject + " " +
            extract_text(email.bodyHtml, email.bodyText) + " " +
            " ".join(email.urls))
    X = vectorizer.transform([text])
    proba = model.predict_proba(X)[0][1]
    label = "phishing" if proba >= 0.5 else "legitimate"
    return {"label": label, "score": round(float(proba), 4)}
