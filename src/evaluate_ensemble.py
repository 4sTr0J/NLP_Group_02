import sys
from pathlib import Path

project_root = Path(r"D:\NLP project").resolve()
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

import pandas as pd
import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pygments.lexers import guess_lexer
from pygments.util import ClassNotFound
from feature_engineering import CodeFeatureExtract

model_dir    = project_root / "models" / "codebert"
test_path    = project_root / "data" / "test_data.csv"

# Language detection
_PYGMENTS_TO_LANG = {
    "c":          "c",
    "c++":        "cpp",
    "python":     "python",
    "python 3":   "python",
    "java":       "java",
    "javascript": "javascript",
}

def detect_lang(code: str) -> str:
    try:
        lexer = guess_lexer(code)
        name  = lexer.name.lower()
        return _PYGMENTS_TO_LANG.get(name, "c")
    except ClassNotFound:
        return "c"
    except Exception:
        return "c"

def main():
    print("Loading test data...", flush=True)
    df = pd.read_csv(test_path)
    print(f"Total test samples available: {len(df)}")
    
    # Evaluate on the full test set
    total_samples = len(df)
    print(f"Evaluating on all {total_samples} samples.")
    df_subset = df.copy()
    
    print("Loading models, please wait...", flush=True)
    bundle     = joblib.load(project_root / "models" / "xgboost_model.pkl")
    xgb_model  = bundle["model"]
    extractor  = bundle["extractor"]

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer  = AutoTokenizer.from_pretrained(model_dir)
    bert_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    bert_model.to(device)
    bert_model.eval()
    print("Models loaded.\n", flush=True)

    xgb_preds = []
    bert_preds = []
    ensemble_preds = []
    targets = df_subset["target"].tolist()

    print("Running predictions...", flush=True)
    for idx, row in df_subset.iterrows():
        code = row["code"]
        
        # XGBoost Prediction
        lang = detect_lang(code)
        extractor.lang = [lang]
        x_features = extractor.transform([code])
        xgb_prediction = int(xgb_model.predict(x_features)[0])
        xgb_preds.append(xgb_prediction)
        
        # CodeBERT Prediction
        inputs = tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs         = bert_model(**inputs)
            probabilities   = torch.softmax(outputs.logits, dim=1)
            bert_prediction = int(torch.argmax(probabilities).item())
            bert_preds.append(bert_prediction)
            
        # Ensemble Prediction
        if xgb_prediction == 1 or bert_prediction == 1:
            ensemble_prediction = 1
        else:
            ensemble_prediction = 0
        ensemble_preds.append(ensemble_prediction)

    # Metrics
    for name, preds in [("XGBoost", xgb_preds), ("CodeBERT", bert_preds), ("Ensemble (Union)", ensemble_preds)]:
        print(f"\n=== {name} Performance ===")
        print(f"Accuracy : {accuracy_score(targets, preds):.4f}")
        print(f"Precision: {precision_score(targets, preds, zero_division=0):.4f}")
        print(f"Recall   : {recall_score(targets, preds, zero_division=0):.4f}")
        print(f"F1 Score : {f1_score(targets, preds, zero_division=0):.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(targets, preds))

if __name__ == "__main__":
    main()
