import ast
import os
import joblib
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences



# 1. Load Test Dataset

print("Loading test dataset...")

test = pd.read_csv("data/test_tokenized.csv")

test["tokens"] = test["tokens"].apply(ast.literal_eval)

test_text = test["tokens"].apply(lambda x: " ".join(x))

y_test = test["target"]

print(f"Test records: {len(test)}")


# 2. RANDOM FOREST EVALUATION

print("\n======================================")
print("       RANDOM FOREST EVALUATION")
print("======================================")

# Load TF-IDF vectorizer
vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

# Load trained Random Forest model
rf_model = joblib.load(
    "models/random_forest.pkl"
)

# Transform test data using the existing TF-IDF vectorizer
X_test_rf = vectorizer.transform(test_text)

# Make predictions
rf_predictions = rf_model.predict(X_test_rf)

# Calculate metrics
rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

rf_precision = precision_score(
    y_test,
    rf_predictions,
    zero_division=0
)

rf_recall = recall_score(
    y_test,
    rf_predictions,
    zero_division=0
)

rf_f1 = f1_score(
    y_test,
    rf_predictions,
    zero_division=0
)

print(f"Accuracy : {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall   : {rf_recall:.4f}")
print(f"F1-Score : {rf_f1:.4f}")

rf_cm = confusion_matrix(y_test, rf_predictions)
print("\nRandom Forest Confusion Matrix:")
print(rf_cm)

os.makedirs("reports", exist_ok=True)
rf_display = ConfusionMatrixDisplay(
    confusion_matrix=rf_cm,
    display_labels=["Benign", "Vulnerable"]
)
rf_display.plot(cmap=plt.cm.Blues)
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig("reports/random_forest_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()
print("\nRandom Forest confusion matrix saved to reports/random_forest_confusion_matrix.png")

print("\nRandom Forest Classification Report:")
print(
    classification_report(
        y_test,
        rf_predictions,
        zero_division=0
    )
)


# 3. CNN EVALUATION

print("\n======================================")
print("           CNN EVALUATION")
print("======================================")

# Load tokenizer
tokenizer = joblib.load(
    "models/tokenizer.pkl"
)

# Load trained CNN model
cnn_model = load_model(
    "models/cnn_model.keras"
)

# Convert test text into sequences
X_test_cnn = tokenizer.texts_to_sequences(
    test_text
)

# IMPORTANT:
# Use the same MAX_LENGTH that was used during CNN training.
MAX_LENGTH = 300

# Pad sequences
X_test_cnn = pad_sequences(
    X_test_cnn,
    maxlen=MAX_LENGTH
)

# Make predictions
cnn_probabilities = cnn_model.predict(
    X_test_cnn,
    verbose=0
)

# Convert probabilities to binary predictions
cnn_predictions = (
    cnn_probabilities >= 0.5
).astype(int).flatten()


# Calculate metrics
cnn_accuracy = accuracy_score(
    y_test,
    cnn_predictions
)

cnn_precision = precision_score(
    y_test,
    cnn_predictions,
    zero_division=0
)

cnn_recall = recall_score(
    y_test,
    cnn_predictions,
    zero_division=0
)

cnn_f1 = f1_score(
    y_test,
    cnn_predictions,
    zero_division=0
)

print(f"Accuracy : {cnn_accuracy:.4f}")
print(f"Precision: {cnn_precision:.4f}")
print(f"Recall   : {cnn_recall:.4f}")
print(f"F1-Score : {cnn_f1:.4f}")

cnn_cm = confusion_matrix(y_test, cnn_predictions)
print("\nCNN Confusion Matrix:")
print(cnn_cm)

cnn_display = ConfusionMatrixDisplay(
    confusion_matrix=cnn_cm,
    display_labels=["Benign", "Vulnerable"]
)
cnn_display.plot(cmap=plt.cm.Blues)
plt.title("CNN Confusion Matrix")
plt.tight_layout()
plt.savefig("reports/cnn_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()
print("\nCNN confusion matrix saved to reports/cnn_confusion_matrix.png")

print("\nCNN Classification Report:")
print(
    classification_report(
        y_test,
        cnn_predictions,
        zero_division=0
    )
)


# 4. MODEL COMPARISON

print("\n======================================")
print("          MODEL COMPARISON")
print("======================================")

results = pd.DataFrame({
    "Model": [
        "Random Forest",
        "CNN"
    ],

    "Accuracy": [
        rf_accuracy,
        cnn_accuracy
    ],

    "Precision": [
        rf_precision,
        cnn_precision
    ],

    "Recall": [
        rf_recall,
        cnn_recall
    ],

    "F1-Score": [
        rf_f1,
        cnn_f1
    ]
})

print("\n")
print(results.to_string(index=False))


# 5. Determine Best Model

best_model = results.loc[
    results["F1-Score"].idxmax(),
    "Model"
]

print("\n======================================")
print("             BEST MODEL")
print("======================================")

print(
    f"Best model based on F1-Score: {best_model}"
)


# 6. Save Evaluation Results

results.to_csv(
    "models/model_comparison.csv",
    index=False
)

print("\nEvaluation results saved to:")
print("models/model_comparison.csv")

print("\nModel evaluation completed successfully!")

# Load model comparison results
results = pd.read_csv("models/model_comparison.csv")

# Accuracy comparison of graphs between Random Forest and CNN

plt.figure(figsize=(8, 5))

plt.bar(
    results["Model"],
    results["Accuracy"]
)

plt.title("Random Forest vs CNN - Accuracy")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)

# Display values on top of bars
for i, value in enumerate(results["Accuracy"]):
    plt.text(
        i,
        value + 0.02,
        f"{value:.2%}",
        ha="center"
    )

plt.tight_layout()
plt.show()

# Precision, Recall and F1-Score comparison

metrics = [
    "Precision",
    "Recall",
    "F1-Score"
]

x = range(len(results["Model"]))
width = 0.25

plt.figure(figsize=(10, 6))

for i, metric in enumerate(metrics):
    values = results[metric]

    positions = [
        position + (i - 1) * width
        for position in x
    ]

    plt.bar(
        positions,
        values,
        width=width,
        label=metric
    )

    # Display percentage values
    for position, value in zip(positions, values):
        plt.text(
            position,
            value + 0.015,
            f"{value:.2%}",
            ha="center",
            fontsize=9
        )

plt.title("Random Forest vs CNN - Precision, Recall and F1-Score")
plt.xlabel("Model")
plt.ylabel("Score")
plt.xticks(
    list(x),
    results["Model"]
)

plt.ylim(0, 1)
plt.legend()

plt.tight_layout()
plt.show()

# Overall performance comparison

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score"
]

x = range(len(metrics))
width = 0.35

plt.figure(figsize=(11, 6))

rf_values = results.loc[
    results["Model"] == "Random Forest",
    metrics
].values.flatten()

cnn_values = results.loc[
    results["Model"] == "CNN",
    metrics
].values.flatten()

plt.bar(
    [i - width / 2 for i in x],
    rf_values,
    width=width,
    label="Random Forest"
)

plt.bar(
    [i + width / 2 for i in x],
    cnn_values,
    width=width,
    label="CNN"
)

# Add percentage values
for i, value in enumerate(rf_values):
    plt.text(
        i - width / 2,
        value + 0.015,
        f"{value:.2%}",
        ha="center",
        fontsize=9
    )

for i, value in enumerate(cnn_values):
    plt.text(
        i + width / 2,
        value + 0.015,
        f"{value:.2%}",
        ha="center",
        fontsize=9
    )

plt.title("Random Forest vs CNN - Overall Performance")
plt.xlabel("Evaluation Metric")
plt.ylabel("Score")

plt.xticks(
    list(x),
    metrics
)

plt.ylim(0, 1)
plt.legend()

plt.tight_layout()
plt.show()
