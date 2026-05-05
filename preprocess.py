import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 🔥 Custom slang dictionary
slang_dict = {
    "bakwas": "bad",
    "faltu": "bad",
    "bekar": "bad",
    "ganda": "bad",
    "acha": "good",
    "mast": "good"
}

def clean_text(text):
    text = text.lower()

    # Replace slang words
    for word, replacement in slang_dict.items():
        text = text.replace(word, replacement)

    # Remove special characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # Tokenize
    words = text.split()

    # Remove stopwords
    words = [w for w in words if w not in stop_words and len(w) > 2]

    return " ".join(words)