from sklearn.metrics import accuracy_score, f1_score, classification_report
import os


def generate_report(y_true, y_pred):
    from sklearn.metrics import classification_report
    import os
    
    os.makedirs("outputs/reports", exist_ok=True)
    
    report = classification_report(y_true, y_pred)
    
    with open("outputs/reports/classification_report.txt", "w") as f:
        f.write(report)

from sklearn.metrics import accuracy_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("Accuracy:", acc) 
    return acc  



