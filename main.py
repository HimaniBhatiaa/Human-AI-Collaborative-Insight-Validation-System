import uvicorn

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
    
from fastapi import FastAPI
import joblib
import sys
import os

# fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from pydantic import BaseModel
import joblib
from src.preprocess import clean_text

app = FastAPI(title="Sentiment Analysis API")

# load model
model = joblib.load("models/best_model.pkl")
tfidf = joblib.load("models/tfidf.pkl")

# request body
class InputText(BaseModel):
    text: str

# home route
@app.get("/")
def home():
    return {"message": "API is running 🚀"}

# predict route
@app.post("/predict")
def predict(data: InputText):
    text = data.text
    print("Received:", text)

    cleaned = clean_text(text)
    vector = tfidf.transform([cleaned])

    pred = model.predict(vector)[0]
    prob = model.predict_proba(vector)[0].max()

    label = "Positive" if pred == 1 else "Negative"

    return {
        "prediction": label,
        "confidence": float(prob)
    }