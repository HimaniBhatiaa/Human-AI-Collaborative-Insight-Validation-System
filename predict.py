import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import joblib
from src.preprocess import clean_text
from feedback import save_feedback

print("🚀 Loading model...")

model = joblib.load("models/best_model.pkl")
tfidf = joblib.load("models/tfidf.pkl")

print("✅ Model Loaded!")

text = input("Enter review: ")

clean = clean_text(text)
vector = tfidf.transform([clean])

pred = int(model.predict(vector)[0])
prob = model.predict_proba(vector)[0]

label = "😊 Positive Review" if pred == 1 else "😡 Negative Review"
confidence = prob[pred]

print(f"\nPrediction: {label}")
print(f"Confidence: {confidence*100:.2f}%") 

# ✅ Save feedback
save_feedback(text, label)