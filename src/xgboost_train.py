import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, confusion_matrix

train = pd.read_csv(r'G:\My Drive\NLP project\data\train_data.csv')
test = pd.read_csv(r'G:\My Drive\NLP project\data\test_data.csv')

vectorizer = joblib.load(r'G:\My Drive\NLP project\models\tfidf_vectorizer.pkl')

x_train = vectorizer.transform(train["code"])
y_train = train["target"]

x_test = vectorizer.transform(test["code"])
y_test = test["target"]

#trees, depth, learning rate
model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)

model.fit(x_train, y_train)

joblib.dump(model, r"G:\My Drive\NLP project\models\xgboost_model.pkl")

#evalutaion
pred = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred, average = "weighted"))
print("Recall:", recall_score(y_test, pred, average = "weighted"))
print("F1 Score:", f1_score(y_test, pred, average = "weighted"))

print("Confusion Matrix:\n", confusion_matrix(y_test, pred))