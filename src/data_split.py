import pandas as pd

try:
    from IPython.display import display
except Exception:
    # Fallback if IPython is not available (e.g., running as a script)
    def display(x):
        # simple fallback: print first few rows for DataFrame, else print repr
        try:
            print(x.head())
        except Exception:
            print(repr(x))

#Load Dataset
df = pd.read_csv("data/vulnerabilities.csv")
display(df)