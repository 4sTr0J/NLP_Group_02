import pandas as pd
import ast
import joblib

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenized datasets
train = pd.read_csv("data/train_tokenized.csv")
test = pd.read_csv("data/test_tokenized.csv")

# Convert strings back to lists
train["tokens"] = train["tokens"].apply(ast.literal_eval)
test["tokens"] = test["tokens"].apply(ast.literal_eval)

# Join tokens into strings
train_text = train["tokens"].apply(lambda x: " ".join(x))
test_text = test["tokens"].apply(lambda x: " ".join(x))

# Create tokenizer
tokenizer = Tokenizer()

# Fit only on training data
tokenizer.fit_on_texts(train_text)

# Convert to integer sequences
X_train = tokenizer.texts_to_sequences(train_text)
X_test = tokenizer.texts_to_sequences(test_text)

# Pad sequences
MAX_LENGTH = 300

X_train = pad_sequences(X_train, maxlen=MAX_LENGTH)
X_test = pad_sequences(X_test, maxlen=MAX_LENGTH)

# Labels
y_train = train["target"]
y_test = test["target"]

# Save tokenizer
joblib.dump(tokenizer, "models/tokenizer.pkl")

print("Training shape :", X_train.shape)
print("Testing shape  :", X_test.shape)