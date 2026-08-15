from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "diversevul_stratified_10000.csv"
)

OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "common"

TRAIN_FILE = OUTPUT_DIRECTORY / "common_train.csv"
VALIDATION_FILE = OUTPUT_DIRECTORY / "common_validation.csv"
TEST_FILE = OUTPUT_DIRECTORY / "common_test.csv"


def calculate_sha256(file_path: Path) -> str:
    """Calculate a SHA-256 hash so members can verify identical files."""
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(block)

    return sha256.hexdigest()


def show_distribution(name: str, dataframe: pd.DataFrame) -> None:
    """Display size and target-class distribution."""
    print(f"\n{name}")
    print("-" * 50)
    print("Records:", len(dataframe))
    print(dataframe["target"].value_counts().sort_index())


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    dataframe = pd.read_csv(INPUT_FILE).reset_index(drop=True)

    required_columns = {"func", "target"}

    if not required_columns.issubset(dataframe.columns):
        raise ValueError(
            f"Dataset must contain these columns: {required_columns}"
        )

    dataframe["func"] = dataframe["func"].astype(str)
    dataframe["target"] = dataframe["target"].astype(int)

    # Preserve the original row identity for overlap verification.
    if "sample_id" not in dataframe.columns:
        dataframe.insert(0, "sample_id", dataframe.index)

    train_data, temporary_data = train_test_split(
        dataframe,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=dataframe["target"],
    )

    validation_data, test_data = train_test_split(
        temporary_data,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temporary_data["target"],
    )

    train_data = train_data.reset_index(drop=True)
    validation_data = validation_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    assert len(train_data) == 7000
    assert len(validation_data) == 1500
    assert len(test_data) == 1500

    train_ids = set(train_data["sample_id"])
    validation_ids = set(validation_data["sample_id"])
    test_ids = set(test_data["sample_id"])

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    train_data.to_csv(TRAIN_FILE, index=False)
    validation_data.to_csv(VALIDATION_FILE, index=False)
    test_data.to_csv(TEST_FILE, index=False)

    print("\nCommon dataset split created successfully.")
    print("Random seed:", RANDOM_SEED)

    show_distribution("Training dataset", train_data)
    show_distribution("Validation dataset", validation_data)
    show_distribution("Testing dataset", test_data)

    print("\nFile verification hashes")
    print("-" * 50)
    print("Train:", calculate_sha256(TRAIN_FILE))
    print("Validation:", calculate_sha256(VALIDATION_FILE))
    print("Test:", calculate_sha256(TEST_FILE))

    print("\nFiles created:")
    print(TRAIN_FILE)
    print(VALIDATION_FILE)
    print(TEST_FILE)


if __name__ == "__main__":
    main()