import pandas as pd
from pathlib import Path

project_root = Path(r"D:\NLP project").resolve()
train_path = project_root / "data" / "train_data.csv"

def main():
    print("Loading current imbalanced train_data.csv...")
    df = pd.read_csv(train_path)
    print(f"Loaded shape: {df.shape}")
    print(df["target"].value_counts())
    
    # Separate classes
    vuln_df = df[df["target"] == 1]
    safe_df = df[df["target"] == 0]
    
    n_vuln = len(vuln_df)
    print(f"Number of vulnerable samples: {n_vuln}")
    
    # Under-sample the safe class to match the vulnerable count (1:1 balance)
    print("Under-sampling the safe class...")
    balanced_safe_df = safe_df.sample(n=n_vuln, random_state=42)
    
    # Merge and shuffle
    balanced_df = pd.concat([vuln_df, balanced_safe_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"New balanced dataset shape: {balanced_df.shape}")
    print(balanced_df["target"].value_counts())
    
    # Save back to train_data.csv
    balanced_df.to_csv(train_path, index=False)
    print("Successfully balanced train_data.csv to a 1:1 class ratio!")

if __name__ == "__main__":
    main()
