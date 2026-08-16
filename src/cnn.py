import ast
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenized datasets
train = pd.read_csv("data/train_tokenized.csv")
test = pd.read_csv("data/test_tokenized.csv")

# Convert strings back to lists
train["tokens"] = train["tokens"].apply(ast.literal_eval)
test["tokens"] = test["tokens"].apply(ast.literal_eval)

# Convert token lists into strings
train_text = train["tokens"].apply(lambda x: " ".join(x))
test_text = test["tokens"].apply(lambda x: " ".join(x))

# Labels
y_train = train["target"]
y_test = test["target"]

# Load tokenizer
tokenizer = joblib.load("models/tokenizer.pkl")

# Convert text into sequences
X_train = tokenizer.texts_to_sequences(train_text)
X_test = tokenizer.texts_to_sequences(test_text)

# Padding
MAX_LENGTH = 300

X_train = pad_sequences(X_train, maxlen=MAX_LENGTH)
X_test = pad_sequences(X_test, maxlen=MAX_LENGTH)

# Vocabulary size
vocab_size = len(tokenizer.word_index)

# CNN model
model = Sequential()

model.add(
    Embedding(
        input_dim=vocab_size + 1,
        output_dim=128,
        input_length=MAX_LENGTH
    )
)

model.add(
    Conv1D(
        filters=128,
        kernel_size=5,
        activation="relu"
    )
)

model.add(GlobalMaxPooling1D())

model.add(Dense(64, activation="relu"))

model.add(Dense(1, activation="sigmoid"))

# Compile
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=32
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)

# Generate confusion matrix
cnn_probabilities = model.predict(X_test, verbose=0).ravel()
cnn_predictions = (cnn_probabilities >= 0.5).astype(int)
cnn_confusion_matrix = confusion_matrix(y_test, cnn_predictions)

print("\nTest Accuracy :", accuracy)
print("\nCNN Confusion Matrix:")
print(cnn_confusion_matrix)

os.makedirs("reports", exist_ok=True)
cm_display = ConfusionMatrixDisplay(
    confusion_matrix=cnn_confusion_matrix,
    display_labels=["Benign", "Vulnerable"]
)
cm_display.plot(cmap=plt.cm.Blues)
plt.title("CNN Confusion Matrix")
plt.tight_layout()
plt.savefig("reports/cnn_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nConfusion matrix saved to reports/cnn_confusion_matrix.png")

# Save model
model.save("models/cnn_model.keras")

print("\nCNN model saved successfully.")
