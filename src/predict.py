import sys
from pathlib import Path

print("Loading models, please wait...", flush=True)

import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pygments.lexers import guess_lexer
from pygments.util import ClassNotFound
from feature_engineering import CodeFeatureExtract

project_root = Path(r"G:\My Drive\NLP project").resolve()
model_dir    = project_root / "models" / "codebert"   # FIX: use Path object, not a raw string with forward slashes

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


# Load models
bundle     = joblib.load(project_root / "models" / "xgboost_model.pkl")
xgb_model  = bundle["model"]
extractor  = bundle["extractor"]

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FIX: pass Path object so transformers treats it as a local directory, not a HuggingFace repo ID
tokenizer  = AutoTokenizer.from_pretrained(model_dir)
bert_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
bert_model.to(device)
bert_model.eval()

print("Models loaded.\n", flush=True)

# ── Scan loop ─────────────────────────────────────────────────────────────────
while True:
    print("=" * 60)
    print("SOURCE CODE VULNERABILITY DETECTION")
    print("=" * 60)
    print("Paste your source code below.")
    print("When done, type END on a new line and press Enter.")
    print("Type EXIT instead to quit.\n")

    lines = []
    while True:
        line = input()
        if line.strip() == "EXIT":
            print("Exiting. Goodbye!")
            sys.exit(0)
        if line.strip() == "END":
            break
        lines.append(line)

    code = "\n".join(lines)

    if not code.strip():
        print("No code entered. Please try again.\n")
        continue

    # XGBoost prediction
    lang = detect_lang(code)
    extractor.lang = [lang]
    x_features = extractor.transform([code])

    xgb_prediction  = xgb_model.predict(x_features)[0]
    xgb_probability = xgb_model.predict_proba(x_features)[0]
    xgb_confidence  = max(xgb_probability) * 100

    # CodeBERT prediction
    inputs = tokenizer(
        code,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs         = bert_model(**inputs)
        probabilities   = torch.softmax(outputs.logits, dim=1)
        bert_prediction = torch.argmax(probabilities).item()
        bert_confidence = torch.max(probabilities).item() * 100

    # Display results
    print("\n" + "=" * 60)
    print(f"Detected language : {lang.upper()}")

    print("\nXGBoost")
    print("-" * 25)
    print(f"Prediction : {'Vulnerable' if xgb_prediction else 'Safe'}")
    print(f"Confidence : {xgb_confidence:.2f}%")

    print("\nCodeBERT")
    print("-" * 25)
    print(f"Prediction : {'Vulnerable' if bert_prediction else 'Safe'}")
    print(f"Confidence : {bert_confidence:.2f}%")

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("-" * 60)

    if xgb_prediction == 1 and bert_prediction == 1:
        final = "HIGH RISK"
    elif xgb_prediction == 0 and bert_prediction == 0:
        final = "SAFE"
    else:
        final = "MANUAL REVIEW REQUIRED"

    print(f"Overall Result : {final}")
    print("=" * 60)

    print("\nRecommendation")
    print("-" * 60)

    if final == "HIGH RISK":
        print("✓ Vulnerabilities detected by both models.")
        print("✓ Review unsafe memory operations.")
        print("✓ Validate all user inputs.")
        print("✓ Avoid strcpy(), gets(), sprintf().")
        print("✓ Use strncpy(), fgets(), snprintf() instead.")
    elif final == "SAFE":
        print("✓ No vulnerability detected.")
        print("✓ Continue following secure coding practices.")
    else:
        print("✓ The two models disagree.")
        print("✓ Manual code review is recommended.")

    print("\n")