import pandas as pd
from pathlib import Path

project_root = Path(r"D:\NLP project").resolve()
train_path = project_root / "data" / "train_data.csv"
vuln_path = project_root / "data" / "vulnerabilities.csv"

import re

def remove_comments(text):
    if not isinstance(text, str):
        return ""
    # Remove single-line comments
    code = re.sub(r'//.*', '', text)
    # Remove multi-line comments
    code = re.sub(r'/\*[\s\S]*?\*/', '', code)
    # Normalize spaces/tabs per line, preserving newline structure
    lines = []
    for line in code.splitlines():
        norm_line = re.sub(r'[ \t]+', ' ', line).strip()
        if norm_line:
            lines.append(norm_line)
    return "\n".join(lines)

def main():
    print("Loading current train_data.csv...")
    train_df = pd.read_csv(train_path)
    print(f"Current train_data shape: {train_df.shape}")
    print(train_df["target"].value_counts())
    
    # Store existing codes in a set for fast lookup
    existing_codes = set(train_df["code"].dropna())
    
    print("\nReading vulnerabilities.csv in chunks...")
    safe_samples = []
    
    # Read only required columns to save memory
    chunk_size = 50000
    for chunk in pd.read_csv(vuln_path, usecols=["func", "target"], chunksize=chunk_size):
        # Filter for Safe (target = 0)
        safe_chunk = chunk[chunk["target"] == 0]
        
        for _, row in safe_chunk.iterrows():
            code = remove_comments(str(row["func"]))
            if code and code not in existing_codes:
                safe_samples.append({"code": code, "target": 0})
                # Add to set to prevent duplicate extraction
                existing_codes.add(code)
                
        # Limit the extraction to 25,000 new diverse safe samples
        if len(safe_samples) >= 25000:
            break
            
    print(f"Extracted {len(safe_samples)} new unique safe samples.")
    
    # Create DataFrame and append
    new_safe_df = pd.DataFrame(safe_samples[:25000])
    updated_train_df = pd.concat([train_df, new_safe_df], ignore_index=True)
    
    print(f"\nNew train_data shape: {updated_train_df.shape}")
    print(updated_train_df["target"].value_counts())
    
    # Save back to train_data.csv
    updated_train_df.to_csv(train_path, index=False)
    print("Successfully updated train_data.csv!")

if __name__ == "__main__":
    main()
