from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import os

def plot_top_words(df, n=20):

    # create folder automatically if not exists
    os.makedirs("outputs/plots", exist_ok=True)

    text = " ".join(df['review'])
    words = text.split()

    word_counts = Counter(words)
    common_words = word_counts.most_common(n)

    words = [w[0] for w in common_words]
    counts = [w[1] for w in common_words]

    plt.figure(figsize=(10,5))
    plt.bar(words, counts)

    plt.xticks(rotation=45)
    plt.title("Top Words in Dataset")
    plt.xlabel("Words")
    plt.ylabel("Frequency")

    path = "outputs/plots/top_words.png"

    plt.savefig(path)
    plt.close()   # no popup needed (like your other plots)

    print(f"✅ Top words plot saved at: {path}")



def plot_wordcloud(df):
    os.makedirs("outputs/plots", exist_ok=True)

    text = " ".join(df['review'])

    wc = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")

    path = "outputs/plots/wordcloud.png"
    plt.savefig(path)
    plt.close()

    print(f"✅ Wordcloud saved at: {path}")

def plot_top_words_by_sentiment(df, sentiment=1, n=15):

    
    
    label = "Positive" if sentiment == 1 else "Negative"

    text = " ".join(df[df['sentiment'] == sentiment]['review'])
    words = text.split()

    common = Counter(words).most_common(n)

    words = [w[0] for w in common]
    counts = [w[1] for w in common]

    plt.figure(figsize=(10,5))
    plt.bar(words, counts)

    plt.xticks(rotation=45)
    plt.title(f"Top {label} Words")
    plt.xlabel("Words")
    plt.ylabel("Frequency")

    path = f"outputs/plots/top_{label.lower()}_words.png"
    plt.savefig(path)
    plt.close()

    print(f"✅ {label} words plot saved")

