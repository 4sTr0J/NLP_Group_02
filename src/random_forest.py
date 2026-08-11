import ast
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Load tokenized datasets
train = pd.read_csv("data/train_tokenized.csv")
test = pd.read_csv("data/test_tokenized.csv")

# Convert token strings back to lists
train["tokens"] = train["tokens"].apply(ast.literal_eval)
test["tokens"] = test["tokens"].apply(ast.literal_eval)

# Convert token lists into strings
train_text = train["tokens"].apply(lambda x: " ".join(x))
test_text = test["tokens"].apply(lambda x: " ".join(x))

# Labels
y_train = train["target"]
y_test = test["target"]

# Load TF-IDF Vectorizer
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Transform datasets
X_train = vectorizer.transform(train_text)
X_test = vectorizer.transform(test_text)

# Build Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Evaluation
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# Save trained model
joblib.dump(rf_model, "models/random_forest.pkl")

print("\nRandom Forest model saved successfully.")