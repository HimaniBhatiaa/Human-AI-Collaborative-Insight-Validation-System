import streamlit as st
import requests
import pandas as pd

# Page config
st.set_page_config(page_title="Sentiment Dashboard", layout="wide")

# Title
st.title("🎬 Human-AI-Collaborative Insight Validation System ")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction"])

# ---------------------------
# PAGE 1: PREDICTION
# ---------------------------
st.header("📝 Predict Sentiment")

# ✅ ADD THIS LINE (IMPORTANT)
text = st.text_area("Enter your review:")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter text")
    else:
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"text": text},
                timeout=5
            )

            if response.status_code == 200:
                result = response.json()

                st.success(f"Prediction: {result['prediction']}")
                st.info(f"Confidence: {result['confidence']*100:.2f}%")

            else:
                st.error(f"Error: {response.status_code}")

        except requests.exceptions.ConnectionError:
            st.error("❌ API not running! Start FastAPI first.")

        except requests.exceptions.Timeout:
            st.error("⏳ API timeout! Try again.")

