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

##drop columns
df = df.drop(columns=['commit_id', 'hash', 'project', 'message'])
display(df)

#Reduced the dataset to 50,000 rows to match the source code requirements.
#Extracting the Top 25,000 Class 0 Rows
#Extracting the Top 25,000 Class 1 Rows
target_0 = df[df['target'] == 0].sort_values(by='size', ascending=False).head(25000)
target_1 = df[df['target'] == 1].sort_values(by='size', ascending=False).head(25000)
combined_data = pd.concat([target_0, target_1])
display(combined_data)

#Remove Comments from the 'func' Column
import re

def remove_comments(text):
    # Remove single-line comments
    code = re.sub(r'//.*', '', text)
    # Remove multi-line comments
    code = re.sub(r'/\*[\s\S]*?\*/', '', code)
    # Remove extra whitespace
    code = re.sub(r'\s+', ' ', code)
    return code.strip()

combined_data['func'] = combined_data['func'].apply(remove_comments)
display(combined_data)


#Final Check
combined_data.info()
combined_data.isnull().sum()

combined_data.to_csv('data/vulnerabilities_50k.csv', index=False)
display(combined_data)