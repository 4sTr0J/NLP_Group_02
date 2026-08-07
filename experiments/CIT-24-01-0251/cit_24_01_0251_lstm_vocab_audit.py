from pathlib import Path
import re

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

RANDOM_SEED = 42

CURRENT_VOCAB_SIZE = 30_000
CANDIDATE_VOCAB_SIZE = 50_000

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "diversevul_stratified_10000.csv"
)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

df = pd.read_csv(DATASET_PATH)

df = (
    df.dropna(subset=["func", "target"])
    .copy()
)

df["func"] = df["func"].astype(str)
df["target"] = df["target"].astype(int)

assert len(df) == 10_000


# ---------------------------------------------------------
# Reproduce the same 70 / 15 / 15 split
# ---------------------------------------------------------

X = df["func"]
y = df["target"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=RANDOM_SEED,
    stratify=y,
)

X_validation, _, y_validation, _ = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_SEED,
    stratify=y_temp,
)

assert len(X_train) == 7000
assert len(X_validation) == 1500


# ---------------------------------------------------------
# Same source-code tokenizer as the LSTM baseline
# ---------------------------------------------------------

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
    """Convert source code into programming-language tokens."""
    return CODE_TOKEN_PATTERN.findall(str(source_code))


def prepare_code_for_lstm(source_code):
    """Convert code tokens into a space-separated string."""
    return " ".join(code_tokenizer(source_code))


# ---------------------------------------------------------
# Prepare training and validation text
# ---------------------------------------------------------

print("Preparing training and validation source code...")

X_train_prepared = X_train.map(
    prepare_code_for_lstm
)

X_validation_prepared = X_validation.map(
    prepare_code_for_lstm
)


# ---------------------------------------------------------
# Fit vocabulary using TRAINING DATA ONLY
# ---------------------------------------------------------

tokenizer = Tokenizer(
    oov_token="<OOV>",
    filters="",
    lower=False,
    split=" ",
)

tokenizer.fit_on_texts(
    X_train_prepared
)

word_index = tokenizer.word_index

print("\nVocabulary audit")
print("=" * 55)

print(
    f"Total discovered training vocabulary: "
    f"{len(word_index):,}"
)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def token_status(token, vocab_limit):
    """
    Return True if the token would be represented directly
    under the specified vocabulary limit.

    Keras keeps token indices strictly below num_words.
    """
    token_index = word_index.get(token)

    if token_index is None:
        return False

    return token_index < vocab_limit


def audit_texts(texts, vocab_limit):
    total_tokens = 0
    represented_tokens = 0
    oov_tokens = 0

    for text in texts:
        tokens = text.split()

        for token in tokens:
            total_tokens += 1

            if token_status(
                token,
                vocab_limit,
            ):
                represented_tokens += 1
            else:
                oov_tokens += 1

    coverage = (
        represented_tokens / total_tokens
        if total_tokens
        else 0.0
    )

    oov_rate = (
        oov_tokens / total_tokens
        if total_tokens
        else 0.0
    )

    return {
        "total_tokens": total_tokens,
        "represented_tokens": represented_tokens,
        "oov_tokens": oov_tokens,
        "coverage": coverage,
        "oov_rate": oov_rate,
    }


# ---------------------------------------------------------
# Training coverage
# ---------------------------------------------------------

train_30k = audit_texts(
    X_train_prepared,
    CURRENT_VOCAB_SIZE,
)

train_50k = audit_texts(
    X_train_prepared,
    CANDIDATE_VOCAB_SIZE,
)


# ---------------------------------------------------------
# Validation coverage
# ---------------------------------------------------------

validation_30k = audit_texts(
    X_validation_prepared,
    CURRENT_VOCAB_SIZE,
)

validation_50k = audit_texts(
    X_validation_prepared,
    CANDIDATE_VOCAB_SIZE,
)


# ---------------------------------------------------------
# Vulnerable validation subset
# ---------------------------------------------------------

vulnerable_mask = (
    y_validation.reset_index(drop=True) == 1
)

validation_prepared_reset = (
    X_validation_prepared.reset_index(drop=True)
)

vulnerable_validation_texts = (
    validation_prepared_reset[
        vulnerable_mask
    ]
)

vulnerable_30k = audit_texts(
    vulnerable_validation_texts,
    CURRENT_VOCAB_SIZE,
)

vulnerable_50k = audit_texts(
    vulnerable_validation_texts,
    CANDIDATE_VOCAB_SIZE,
)


# ---------------------------------------------------------
# Report
# ---------------------------------------------------------

def print_result(label, result):
    print(f"\n{label}")
    print("-" * 55)

    print(
        f"Total token occurrences: "
        f"{result['total_tokens']:,}"
    )

    print(
        f"Represented directly: "
        f"{result['represented_tokens']:,}"
    )

    print(
        f"OOV occurrences: "
        f"{result['oov_tokens']:,}"
    )

    print(
        f"Vocabulary coverage: "
        f"{result['coverage'] * 100:.2f}%"
    )

    print(
        f"OOV rate: "
        f"{result['oov_rate'] * 100:.2f}%"
    )


print_result(
    "TRAINING - 30K vocabulary",
    train_30k,
)

print_result(
    "TRAINING - 50K vocabulary",
    train_50k,
)

print_result(
    "VALIDATION - 30K vocabulary",
    validation_30k,
)

print_result(
    "VALIDATION - 50K vocabulary",
    validation_50k,
)

print_result(
    "VULNERABLE VALIDATION - 30K vocabulary",
    vulnerable_30k,
)

print_result(
    "VULNERABLE VALIDATION - 50K vocabulary",
    vulnerable_50k,
)


# ---------------------------------------------------------
# Improvement summary
# ---------------------------------------------------------

overall_recovered = (
    validation_30k["oov_tokens"]
    - validation_50k["oov_tokens"]
)

vulnerable_recovered = (
    vulnerable_30k["oov_tokens"]
    - vulnerable_50k["oov_tokens"]
)

print("\nPotential benefit of 30K -> 50K")
print("=" * 55)

print(
    f"Validation token occurrences recovered: "
    f"{overall_recovered:,}"
)

print(
    f"Vulnerable validation token occurrences recovered: "
    f"{vulnerable_recovered:,}"
)

print(
    "\nHeld-out test data was NOT inspected."
)
