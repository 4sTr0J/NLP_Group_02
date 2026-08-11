import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import ast
import joblib

# Load tokenized datasets
train = pd.read_csv("data/train_tokenized.csv")
test = pd.read_csv("data/test_tokenized.csv")

# Convert string representation of lists back to Python lists
train["tokens"] = train["tokens"].apply(ast.literal_eval)
test["tokens"] = test["tokens"].apply(ast.literal_eval)

# Convert token lists to strings
train_text = train["tokens"].apply(lambda x: " ".join(x))
test_text = test["tokens"].apply(lambda x: " ".join(x))

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Fit only on training data
X_train = vectorizer.fit_transform(train_text)

# Transform testing data
X_test = vectorizer.transform(test_text)

# Labels
y_train = train["target"]
y_test = test["target"]

# Save vectorizer
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("Training shape :", X_train.shape)
print("Testing shape  :", X_test.shape)