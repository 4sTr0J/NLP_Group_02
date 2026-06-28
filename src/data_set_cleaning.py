#Load Dataset
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

# Fix: read CSV once (previous code called pd.read_csv twice)
df = pd.read_csv("data/vulnerabilities.csv")
display(df)

#Check Dataset Information
df.info()
print("***************************************")
df.describe()

#Check Missing Values
df.isnull().sum()

df.dropna(subset=["cwe"], inplace=True)

#Remove Duplicate Records
df.duplicated().sum()
df.drop_duplicates(inplace=True)
#print(df)

#Remove Unnecessary Spaces
text_columns = ['func','cwe','project','message']

for col in text_columns:
    df[col] = df[col].astype(str).str.strip()

display(df.head())

#Convert Data Types
df['target'] = df['target'].astype(int)
display(df)

#Check Unique Target Values
df['target'].value_counts()

#Remove Special Characters from Text Columns
import re

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text))
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

for col in ['func','cwe','project','message']:
    df[col] = df[col].apply(clean_text)

#Encode Categorical Features
for col in ['project','cwe','func']:
    df[col], _ = pd.factorize(df[col])

display(df)

#Final Check
df.info()
df.isnull().sum()

df_50k = df.sample(n=50000, random_state=42)

print(df_50k.shape)

df_50k.to_csv('data/vulnerabilities_50k.csv', index=False)
display(df_50k)