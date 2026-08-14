import sys
from pathlib import Path

project_root = Path(r"D:\NLP project").resolve()
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from feature_engineering import CodeFeatureExtract
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
)
from pygments.lexers import guess_lexer
from pygments.util import ClassNotFound

# Language detection
from feature_engineering import detect_lang


# Load data
train = pd.read_csv(r'D:\NLP project\data\train_data.csv')
test  = pd.read_csv(r'D:\NLP project\data\test_data.csv')

# Detect language for each sample
if "lang" not in train.columns:
    print("Detecting languages for train set...")
    train["lang"] = train["code"].apply(detect_lang)
if "lang" not in test.columns:
    print("Detecting languages for test set...")
    test["lang"] = test["code"].apply(detect_lang)

print("Language distribution (train):\n", train["lang"].value_counts())
print("Language distribution (test):\n",  test["lang"].value_counts())

# Feature engineering: fit on train ONLY
train_langs = train["lang"].tolist()
test_langs  = test["lang"].tolist()

extractor = CodeFeatureExtract(
    lang=train_langs,
    max_ngram_features=20000,
)

x_train = extractor.fit_transform(train["code"])
y_train = train["target"]

# BUG FIX: instead of mutating extractor.lang (fragile), create a fresh extractor
# that shares the fitted vectorizer_ so the same vocabulary is applied to test data.
test_extractor = CodeFeatureExtract(lang=test_langs, max_ngram_features=20000)
test_extractor.vectorizer_    = extractor.vectorizer_
test_extractor.feature_names_ = extractor.feature_names_

x_test = test_extractor.transform(test["code"])
y_test = test["target"]

# Handle class imbalance
scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
print(f"scale_pos_weight: {scale_pos_weight:.4f}")

# Train
model = XGBClassifier(
    n_estimators=800,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="aucpr",
    random_state=42,
)

model.fit(x_train, y_train)

# BUG FIX: predict.py loads this bundle, so the keys must match what predict.py expects.
# Saving both model and extractor together so predict.py can load them as one unit.
joblib.dump(
    {"model": model, "extractor": extractor},
    r"D:\NLP project\models\xgboost_model.pkl",
)
print("Model saved to models/xgboost_model.pkl")

# Evaluation
pred = model.predict(x_test)
prob = model.predict_proba(x_test)[:, 1]

print("\nAccuracy :",  accuracy_score(y_test, pred))
print("Precision:",   precision_score(y_test, pred, average="weighted"))
print("Recall   :",   recall_score(y_test, pred, average="weighted"))
print("F1 Score :",   f1_score(y_test, pred, average="weighted"))
print("ROC-AUC  :",   roc_auc_score(y_test, prob))
print("\nConfusion Matrix:\n",       confusion_matrix(y_test, pred))
print("\nClassification Report:\n",  classification_report(y_test, pred, target_names=["safe", "vulnerable"]))

importances = pd.Series(
    model.feature_importances_, index=extractor.get_feature_names_out()
).sort_values(ascending=False)
print("\nTop 20 features:\n", importances.head(20))