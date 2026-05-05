import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


from src.preprocess import clean_text
from src.models import get_vectorizer, get_models
from src.feature_engineering import plot_top_words

from src.evaluate import evaluate_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

print("🚀 PIPELINE STARTED...")

# ===============================
# STEP 1: LOAD DATA
# ===============================
df = pd.read_csv("data/raw/IMDB Dataset.csv")
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

print("✅ Data Loaded:", df.shape)

# ===============================
# STEP 2: EDA
# ===============================
counts = df['sentiment'].value_counts()

plt.figure()
counts.plot(kind='bar')
plt.title("Sentiment Distribution")
plt.savefig("outputs/plots/sentiment_distribution.png")
plt.close()

# ===============================
# STEP 3: TEXT CLEANING
# ===============================
df['clean_review'] = df['review'].apply(clean_text)

# ===============================
# STEP 4: FEATURE ENGINEERING
# ===============================
tfidf = get_vectorizer()
X = tfidf.fit_transform(df['clean_review'])
y = df['sentiment']

# ===============================
# STEP 5: TRAIN TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# STEP 6: MODEL TRAINING
# ===============================
models = get_models()
results = {}

for name, model in models.items():
    print(f"\n🚀 Training: {name}")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = evaluate_model(model, X_test, y_test) 
    results[name] = acc

# ===============================
# STEP 7: BEST MODEL
# ===============================

print("Results:", results)

results = {k: v for k, v in results.items() if v is not None}

best_name = max(results, key=results.get)


best_model = models[best_name]

print(f"\n🏆 Best Model: {best_name}")



print(f"\n🏆 Best Model: {best_name}")

# ===============================
# STEP 8: CONFUSION MATRIX
# ===============================
y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.savefig("outputs/plots/confusion_matrix.png")
plt.close()

# ===============================
# STEP 9: TOP WORDS PLOT
# ===============================
plot_top_words(df)

# ===============================
# STEP 10: SAVE MODEL
# ===============================
joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(tfidf, "models/tfidf.pkl")

print("✅ Model Saved")

# ===============================
# STEP 11: REPORT
# ===============================
with open("outputs/reports/summary.txt", "w", encoding="utf-8") as f:
    f.write("📊 PROJECT SUMMARY\n")
    f.write(f"Best Model: {best_name}\n")
    f.write(f"Accuracy: {results[best_name]:.4f}\n")

print("📄 Report Generated")

print("🎉 PIPELINE COMPLETED!")

