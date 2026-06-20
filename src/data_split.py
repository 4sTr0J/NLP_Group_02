import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'G:\My Drive\NLP project\data\cleaned_data_of_vulnerabilities.csv')

x = df["func"]
y = df["target"]

#allocate 20% of the data for testing and 80% for training
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42, stratify=y, shuffle=True)

train = pd.DataFrame({
    "code":x_train,
    "target":y_train
})

test = pd.DataFrame({
    "code":x_test,
    "target":y_test 
})

train.to_csv(r'G:\My Drive\NLP project\data\train_data.csv', index=False)
test.to_csv(r'G:\My Drive\NLP project\data\test_data.csv', index=False)
