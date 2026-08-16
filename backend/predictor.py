import os
import re
import joblib
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# Project paths
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# ============================================================
# Model files
# ============================================================

TFIDF_PATH = os.path.join(
    MODELS_DIR,
    "tfidf_vectorizer.pkl"
)

TOKENIZER_PATH = os.path.join(
    MODELS_DIR,
    "tokenizer.pkl"
)

RF_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "random_forest.pkl"
)

CNN_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "cnn_model.keras"
)


# ============================================================
# CNN configuration
# ============================================================

# This MUST match feature_extraction_cnn.py
MAX_LENGTH = 300


# ============================================================
# Load trained models
# ============================================================

print("Loading TF-IDF vectorizer...")
tfidf_vectorizer = joblib.load(TFIDF_PATH)

print("Loading tokenizer...")
tokenizer = joblib.load(TOKENIZER_PATH)

print("Loading Random Forest model...")
random_forest = joblib.load(RF_MODEL_PATH)

print("Loading CNN model...")
cnn_model = load_model(CNN_MODEL_PATH)

print("All models loaded successfully.")


# ============================================================
# EXACT TOKENIZATION USED DURING TRAINING
# ============================================================

def tokenize_code(code):
    """
    Tokenize source code using the exact same
    regular expression used during model training.
    """

    code = str(code)

    tokens = re.findall(
        r"[A-Za-z_]\w*|"
        r"\d+|"
        r"==|!=|<=|>=|&&|\|\||"
        r"[{}()\[\];,=+\-*/<>]",
        code
    )

    return tokens


# ============================================================
# Prepare code
# ============================================================

def prepare_code(code):
    """
    Apply the same tokenization and text conversion
    used during feature extraction.
    """

    tokens = tokenize_code(code)

    text = " ".join(tokens)

    return tokens, text


# ============================================================
# Random Forest Prediction
# ============================================================

def predict_random_forest(code):

    tokens, text = prepare_code(code)

    # IMPORTANT:
    # Do NOT fit the vectorizer again.
    # It was already fitted on the training dataset.
    X = tfidf_vectorizer.transform([text])

    # Prediction
    prediction = random_forest.predict(X)[0]

    # Probability
    probabilities = random_forest.predict_proba(X)[0]

    prediction = int(prediction)

    confidence = float(
        probabilities[prediction]
    )

    if prediction == 1:
        label = "Vulnerable"
    else:
        label = "Safe"

    return {
        "prediction": prediction,
        "label": label,
        "confidence": confidence
    }


# ============================================================
# CNN Prediction
# ============================================================

def predict_cnn(code):

    tokens, text = prepare_code(code)

    # Convert text into integer sequence
    sequence = tokenizer.texts_to_sequences(
        [text]
    )

    # Pad exactly like training
    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH
    )

    # CNN prediction
    raw_prediction = cnn_model.predict(
        padded_sequence,
        verbose=0
    )

    # Convert prediction to a scalar
    probability = float(
        np.asarray(raw_prediction).reshape(-1)[0]
    )

    # Make sure probability is within 0-1
    probability = max(
        0.0,
        min(1.0, probability)
    )

    # Binary classification
    if probability >= 0.5:

        prediction = 1
        label = "Vulnerable"
        confidence = probability

    else:

        prediction = 0
        label = "Safe"
        confidence = 1.0 - probability

    return {
        "prediction": prediction,
        "label": label,
        "confidence": confidence
    }


# ============================================================
# Combined Prediction
# ============================================================

def predict_code(code):

    if not code or not code.strip():

        raise ValueError(
            "Source code cannot be empty."
        )

    # Run both models
    rf_result = predict_random_forest(code)

    cnn_result = predict_cnn(code)

    # ========================================================
    # Overall decision
    # ========================================================

    rf_prediction = rf_result["prediction"]
    cnn_prediction = cnn_result["prediction"]

    if (
        rf_prediction == 1
        and cnn_prediction == 1
    ):

        overall_label = "Vulnerable"

    elif (
        rf_prediction == 0
        and cnn_prediction == 0
    ):

        overall_label = "Safe"

    else:

        overall_label = "Uncertain"

    # ========================================================
    # Return complete result
    # ========================================================

    return {

        "random_forest": rf_result,

        "cnn": cnn_result,

        "overall": {
            "label": overall_label
        }
    }