import pandas as pd
import re

# Load training and testing datasets
train = pd.read_csv("data/train_data.csv")
test = pd.read_csv("data/test_data.csv")


# Function to tokenize source code
def tokenize_code(code):
    code = str(code)

    tokens = re.findall(
        r"[A-Za-z_]\w*|"          # identifiers
        r"\d+|"                   # numbers
        r"==|!=|<=|>=|&&|\|\||"   # multi-character operators
        r"[{}()\[\];,=+\-*/<>]",  # single-character operators/symbols
        code
    )

    return tokens


# Apply tokenization
train["tokens"] = train["code"].apply(tokenize_code)
test["tokens"] = test["code"].apply(tokenize_code)


# Display sample output
print("Training Sample:")
print(train[["code", "tokens"]].head())

print("\nTesting Sample:")
print(test[["code", "tokens"]].head())


# Save tokenized datasets
train.to_csv("data/train_tokenized.csv", index=False)
test.to_csv("data/test_tokenized.csv", index=False)

print("\nTokenization completed successfully.")