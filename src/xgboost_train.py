import pandas as pd
import joblib

from xgboost import XGBClassifier

train = pd.read_csv("data/train_data.csv")

vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

x_train = vectorizer.transform(train["code"])
y_train = train["target"]

#trees, depth, learing rate
model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)

model.fit(x_train, y_train)

joblib.dump(model, "models/xgboost_model.pkl")