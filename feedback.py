def save_feedback(text, prediction):
    with open("outputs/feedback.txt", "a", encoding="utf-8") as f:
        f.write(f"{text} -> {prediction}\n")