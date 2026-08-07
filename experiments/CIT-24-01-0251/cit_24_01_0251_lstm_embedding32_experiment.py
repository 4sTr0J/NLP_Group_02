

# %%

from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from keras import Sequential
from keras.callbacks import EarlyStopping, TerminateOnNaN
from keras.layers import Dense, Dropout, Embedding, LSTM
from keras.utils import pad_sequences

# Kept for compatibility with the existing trained tokenizer.
from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

PROJECT_ROOT = Path.cwd()

if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "diversevul_stratified_10000.csv"
)

print("LSTM development environment ready.")
print("TensorFlow version:", tf.__version__)
print("Project root:", PROJECT_ROOT)
print("Dataset path:", DATA_PATH)
print("Dataset exists:", DATA_PATH.exists())

# %%

# Load the processed dataset for LSTM development.

lstm_df = pd.read_csv(
    DATA_PATH,
    usecols=["func", "target"],
)

lstm_df = lstm_df.dropna(
    subset=["func", "target"],
).copy()

lstm_df["func"] = lstm_df["func"].astype(str)
lstm_df["target"] = lstm_df["target"].astype(int)

print("Dataset loaded successfully.")
print("Dataset shape:", lstm_df.shape)

print("\nClass distribution:")
print(
    lstm_df["target"]
    .value_counts()
    .sort_index()
)

print("\nMissing values:")
print(lstm_df.isna().sum())

print("\nDuplicate source-code functions:")
print(lstm_df["func"].duplicated().sum())

assert len(lstm_df) == 10_000
assert set(lstm_df["target"].unique()) == {0, 1}
assert lstm_df["func"].duplicated().sum() == 0

print("\nLSTM dataset validation completed successfully.")

# %%

# Create the same stratified data split used for the SVM model.
# This allows a fair comparison between SVM and LSTM results.

X = lstm_df["func"]
y = lstm_df["target"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=RANDOM_SEED,
    stratify=y,
)

X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_SEED,
    stratify=y_temp,
)

# Convert the labels into NumPy arrays for TensorFlow.
y_train_array = y_train.to_numpy(dtype=np.float32)
y_validation_array = y_validation.to_numpy(dtype=np.float32)
y_test_array = y_test.to_numpy(dtype=np.float32)

print("LSTM dataset split completed successfully.")

print("\nTraining set:")
print("Records:", len(X_train))
print(y_train.value_counts().sort_index())

print("\nValidation set:")
print("Records:", len(X_validation))
print(y_validation.value_counts().sort_index())

print("\nTesting set:")
print("Records:", len(X_test))
print(y_test.value_counts().sort_index())

print("\nTotal records:")
print(len(X_train) + len(X_validation) + len(X_test))

assert len(X_train) == 7000
assert len(X_validation) == 1500
assert len(X_test) == 1500
assert len(y_train_array) == 7000
assert len(y_validation_array) == 1500
assert len(y_test_array) == 1500

print("\nLSTM split validation completed successfully.")

# %%

# Tokenize source code while preserving programming-language symbols.

import re

CODE_TOKEN_PATTERN = re.compile(
    r"""
    0x[0-9A-Fa-f]+
    |
    [A-Za-z_]\w*
    |
    \d+(?:\.\d+)?
    |
    ==|!=|<=|>=|->|\+\+|--|&&|\|\||<<|>>
    |
    [{}()\[\];,.+\-*/%&|^~!<>=?:]
    """,
    re.VERBOSE,
)


def code_tokenizer(source_code):
    """Convert a source-code function into code-related tokens."""
    return CODE_TOKEN_PATTERN.findall(str(source_code))


def prepare_code_for_lstm(source_code):
    """Convert code tokens into a space-separated token string."""
    return " ".join(code_tokenizer(source_code))


example_code = """
int add_numbers(int a, int b) {
    return a + b;
}
"""

print("Example tokens:")
print(code_tokenizer(example_code))

print("\nPrepared code:")
print(prepare_code_for_lstm(example_code))

# %%

# Prepare source-code text and create integer token sequences.
# The tokenizer is fitted only on training data to prevent data leakage.

MAX_VOCAB_SIZE = 30_000

print("Preparing source-code functions...")

X_train_prepared = X_train.map(prepare_code_for_lstm)
X_validation_prepared = X_validation.map(prepare_code_for_lstm)
X_test_prepared = X_test.map(prepare_code_for_lstm)

lstm_tokenizer = Tokenizer(
    num_words=MAX_VOCAB_SIZE,
    oov_token="<OOV>",
    filters="",
    lower=False,
    split=" ",
)

print("Fitting tokenizer on training data...")

lstm_tokenizer.fit_on_texts(
    X_train_prepared
)

X_train_sequences = lstm_tokenizer.texts_to_sequences(
    X_train_prepared
)

X_validation_sequences = lstm_tokenizer.texts_to_sequences(
    X_validation_prepared
)

X_test_sequences = lstm_tokenizer.texts_to_sequences(
    X_test_prepared
)

train_sequence_lengths = np.array(
    [len(sequence) for sequence in X_train_sequences]
)

print("\nToken sequence creation completed successfully.")

print("\nTotal discovered vocabulary:")
print(len(lstm_tokenizer.word_index))

print("\nMaximum vocabulary used by the model:")
print(MAX_VOCAB_SIZE)

print("\nTraining sequences:")
print(len(X_train_sequences))

print("\nValidation sequences:")
print(len(X_validation_sequences))

print("\nTesting sequences:")
print(len(X_test_sequences))

print("\nTraining sequence-length percentiles:")

for percentile in [50, 75, 90, 95, 99]:
    value = int(
        np.percentile(
            train_sequence_lengths,
            percentile,
        )
    )
    print(f"{percentile}th percentile: {value} tokens")

print("\nMaximum training sequence length:")
print(train_sequence_lengths.max())

print("\nEmpty training sequences:")
print(np.sum(train_sequence_lengths == 0))

assert len(X_train_sequences) == 7000
assert len(X_validation_sequences) == 1500
assert len(X_test_sequences) == 1500

print("\nLSTM sequence validation completed successfully.")

# %%

# Pad or truncate every source-code sequence to a fixed length.
# A length of 600 covers approximately 90% of training functions.

MAX_SEQUENCE_LENGTH = 600

X_train_padded = pad_sequences(
    X_train_sequences,
    maxlen=MAX_SEQUENCE_LENGTH,
    padding="post",
    truncating="post",
    dtype="int32",
)

X_validation_padded = pad_sequences(
    X_validation_sequences,
    maxlen=MAX_SEQUENCE_LENGTH,
    padding="post",
    truncating="post",
    dtype="int32",
)

X_test_padded = pad_sequences(
    X_test_sequences,
    maxlen=MAX_SEQUENCE_LENGTH,
    padding="post",
    truncating="post",
    dtype="int32",
)

# Calculate balanced class weights because vulnerable records
# are much less common than non-vulnerable records.

class_weight_values = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=y_train_array.astype(int),
)

lstm_class_weights = {
    0: float(class_weight_values[0]),
    1: float(class_weight_values[1]),
}

VOCAB_SIZE = min(
    MAX_VOCAB_SIZE,
    len(lstm_tokenizer.word_index) + 1,
)

print("Sequence padding completed successfully.")

print("\nTraining input shape:")
print(X_train_padded.shape)

print("\nValidation input shape:")
print(X_validation_padded.shape)

print("\nTesting input shape:")
print(X_test_padded.shape)

print("\nVocabulary size used:")
print(VOCAB_SIZE)

print("\nMaximum sequence length:")
print(MAX_SEQUENCE_LENGTH)

print("\nClass weights:")
print(lstm_class_weights)

assert X_train_padded.shape == (7000, 600)
assert X_validation_padded.shape == (1500, 600)
assert X_test_padded.shape == (1500, 600)

print("\nLSTM input validation completed successfully.")

# %%

# Build the baseline LSTM vulnerability-classification model.

EMBEDDING_DIM = 32
LSTM_UNITS = 64

lstm_model = Sequential(
    [
        tf.keras.Input(
            shape=(MAX_SEQUENCE_LENGTH,),
            dtype="int32",
            name="source_code_tokens",
        ),
        Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=EMBEDDING_DIM,
            mask_zero=True,
            name="token_embedding",
        ),
        LSTM(
            units=LSTM_UNITS,
            dropout=0.20,
            recurrent_dropout=0.0,
            name="lstm_layer",
        ),
        Dropout(
            rate=0.30,
            name="dropout_layer",
        ),
        Dense(
            units=32,
            activation="relu",
            name="dense_layer",
        ),
        Dropout(
            rate=0.20,
            name="dense_dropout",
        ),
        Dense(
            units=1,
            activation="sigmoid",
            name="vulnerability_prediction",
        ),
    ],
    name="source_code_vulnerability_lstm",
)

lstm_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001,
    ),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(
            name="accuracy",
        ),
        tf.keras.metrics.Precision(
            name="precision",
        ),
        tf.keras.metrics.Recall(
            name="recall",
        ),
        tf.keras.metrics.AUC(
            name="roc_auc",
        ),
        tf.keras.metrics.AUC(
            curve="PR",
            name="pr_auc",
        ),
    ],
)

print("LSTM model created and compiled successfully.\n")

lstm_model.summary()

# %%

# Train the LSTM model using class weights and early stopping.

EPOCHS = 8
BATCH_SIZE = 64

early_stopping = EarlyStopping(
    monitor="val_pr_auc",
    mode="max",
    patience=2,
    restore_best_weights=True,
    verbose=1,
)

terminate_on_nan = tf.keras.callbacks.TerminateOnNaN()

print("Starting LSTM training...")
print("Maximum epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)
print("Class weights:", lstm_class_weights)

lstm_history = lstm_model.fit(
    X_train_padded,
    y_train_array,
    validation_data=(
        X_validation_padded,
        y_validation_array,
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=lstm_class_weights,
    callbacks=[
        early_stopping,
        terminate_on_nan,
    ],
    verbose=1,
)

print("\nLSTM training completed successfully.")
print("Epochs completed:", len(lstm_history.history["loss"]))

# %%

# Visualize LSTM training and validation performance.

history_df = pd.DataFrame(lstm_history.history)

print("Training history:")
print(history_df.round(4))

epochs_completed = range(
    1,
    len(history_df) + 1,
)

plt.figure(figsize=(8, 5))
plt.plot(
    epochs_completed,
    history_df["loss"],
    marker="o",
    label="Training loss",
)
plt.plot(
    epochs_completed,
    history_df["val_loss"],
    marker="o",
    label="Validation loss",
)
plt.xlabel("Epoch")
plt.ylabel("Binary cross-entropy loss")
plt.title("LSTM Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(
    epochs_completed,
    history_df["pr_auc"],
    marker="o",
    label="Training PR-AUC",
)
plt.plot(
    epochs_completed,
    history_df["val_pr_auc"],
    marker="o",
    label="Validation PR-AUC",
)
plt.xlabel("Epoch")
plt.ylabel("PR-AUC")
plt.title("LSTM Training and Validation PR-AUC")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%

# Evaluate the restored LSTM model on validation data
# and select the best decision threshold using validation F1-score.

from sklearn.metrics import precision_recall_curve

print("Generating validation probabilities...")

validation_probabilities = lstm_model.predict(
    X_validation_padded,
    batch_size=BATCH_SIZE,
    verbose=1,
).ravel()

validation_average_precision = average_precision_score(
    y_validation_array,
    validation_probabilities,
)

validation_roc_auc = roc_auc_score(
    y_validation_array,
    validation_probabilities,
)

precision_values, recall_values, threshold_values = precision_recall_curve(
    y_validation_array,
    validation_probabilities,
)

# The final precision and recall values do not have a matching threshold.
threshold_precision = precision_values[:-1]
threshold_recall = recall_values[:-1]

threshold_f1 = np.divide(
    2 * threshold_precision * threshold_recall,
    threshold_precision + threshold_recall,
    out=np.zeros_like(threshold_precision),
    where=(threshold_precision + threshold_recall) != 0,
)

best_threshold_index = np.argmax(threshold_f1)

best_lstm_threshold = threshold_values[best_threshold_index]
best_lstm_precision = threshold_precision[best_threshold_index]
best_lstm_recall = threshold_recall[best_threshold_index]
best_lstm_f1 = threshold_f1[best_threshold_index]

optimized_validation_predictions = (
    validation_probabilities >= best_lstm_threshold
).astype(int)

optimized_validation_accuracy = accuracy_score(
    y_validation_array,
    optimized_validation_predictions,
)

optimized_validation_cm = confusion_matrix(
    y_validation_array,
    optimized_validation_predictions,
)

print("\nLSTM validation threshold tuning completed successfully.")

print(f"\nBest decision threshold: {best_lstm_threshold:.4f}")
print(f"Accuracy: {optimized_validation_accuracy:.4f}")
print(f"Precision: {best_lstm_precision:.4f}")
print(f"Recall: {best_lstm_recall:.4f}")
print(f"F1-score: {best_lstm_f1:.4f}")
print(f"Average precision: {validation_average_precision:.4f}")
print(f"ROC-AUC: {validation_roc_auc:.4f}")

print("\nOptimized validation confusion matrix:")
print(optimized_validation_cm)

# %%

# Embedding-32 LSTM validation experiment ends here.
# Held-out test metrics are intentionally not evaluated.
