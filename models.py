from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

def get_models():
    return {
        "logistic": LogisticRegression(),
        "naive_bayes": MultinomialNB()
    }
from sklearn.feature_extraction.text import TfidfVectorizer

def get_vectorizer():
    return TfidfVectorizer(
        max_features=5000,
        ngram_range=(1,2),
        stop_words='english'
    )
