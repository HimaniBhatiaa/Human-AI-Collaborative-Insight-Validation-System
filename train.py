import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import pandas as pd
import joblib


from src.preprocess import clean_text
from src.models import get_models, get_vectorizer
from src.feature_engineering import (
    plot_top_words,
    plot_wordcloud,
    plot_top_words_by_sentiment
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
import seaborn as sns

print("\n🚀 STEP 1: Loading datasets...")

# IMDB
df_imdb = pd.read_csv("data/raw/IMDB Dataset.csv")

plot_top_words(df_imdb)   

plot_wordcloud(df_imdb)


from src.preprocess import clean_text



# 🔥 ADD THIS LINE (MISSING IN YOUR CODE)
df_imdb['clean_review'] = df_imdb['review'].apply(clean_text)

# DEBUG CHECK
print("Columns:", df_imdb.columns)
print(df_imdb[['review', 'clean_review']].head())



#

# Map sentiment
df_imdb['sentiment'] = df_imdb['sentiment'].map({'positive':1,'negative':0})





print("✅ IMDB Loaded:", df_imdb.shape)

# REVIEWS
df_reviews = pd.read_csv("data/raw/Reviews.csv")

def convert(score):
    if score >= 4:
        return 1
    elif score <= 2:
        return 0
    else:
        return None

df_reviews['sentiment'] = df_reviews['Score'].apply(convert)
df_reviews = df_reviews.dropna()

df_reviews.rename(columns={'Text':'review'}, inplace=True)
df_reviews = df_reviews[['review','sentiment']]

print("✅ Reviews Loaded:", df_reviews.shape)

# COMBINE
df = pd.concat([df_imdb[['review','sentiment']], df_reviews], ignore_index=True)
print("📊 Combined dataset:", df.shape)

# SAMPLE
df = df.sample(10000, random_state=42)

# ---------------- EDA ----------------
print("\n📊 STEP 2: EDA")

print(df['sentiment'].value_counts())

df['sentiment'].value_counts().plot(kind='bar')
plt.title("Sentiment Distribution")
plt.savefig("outputs/plots/sentiment_distribution.png")
plt.close()



print("✅ Plot saved in outputs/plots/")

# ---------------- CLEAN ----------------
print("\n🧹 STEP 3: Cleaning text...")
df['clean_review'] = df['review'].apply(clean_text)

# ---------------- TF-IDF ----------------
print("\n🔢 STEP 4: Vectorizing...")
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['clean_review'])
y = df['sentiment']

# ---------------- SPLIT ----------------
print("\n📂 STEP 5: Train-Test Split...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ---------------- MODELS ----------------
print("\n🤖 STEP 6: Training Models...")

models = {
    "Logistic": LogisticRegression(),
    "NaiveBayes": MultinomialNB(),
    "DecisionTree": DecisionTreeClassifier()
}

results = {}

for name, model in models.items():
    print(f"\n➡ Training {name}...")
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    results[name] = f1

    print(f"{name} → Accuracy: {acc:.4f}, F1: {f1:.4f}")

# ---------------- BEST MODEL ----------------
best_name = max(results, key=results.get)
best_model = models[best_name]

print("\n🏆 Best Model:", best_name)

# ---------------- EVALUATION ----------------
print("\n📊 STEP 7: Confusion Matrix")

y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.savefig("outputs/plots/confusion_matrix.png")
plt.close()

print("✅ Confusion matrix saved")

# ---------------- SAVE ----------------
print("\n💾 STEP 8: Saving model...")
joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(tfidf, "models/tfidf.pkl")

print("✅ Model saved successfully!")