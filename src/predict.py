import argparse
import joblib
import torch

from transformers import ( AutoTokenizer, AutoModelForSequenceClassification)

parser = argparse.ArgumentParser(description="Source Code Vulnerability Detection")

parser.add_argument("file", help="Path to source code file")

args = parser.parse_args()

#read code from file
with open(args.file, "r", encoding="utf-8") as f:
    code = f.read()

#xgboost Prediction
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

xgb_model = joblib.load("models/xgboost/xgb_model.pkl")

x_features = vectorizer.transform([code])

xgb_prediction = xgb_model.predict(x_features)[0]

xgb_probability = xgb_model.predict_proba(x_features)[0]

xgb_confidence = max(xgb_probability) * 100


#bert Prediction
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("models/codebert")

bert_model = AutoModelForSequenceClassification.from_pretrained("models/codebert")

bert_model.to(device)
bert_model.eval()

inputs = tokenizer(
    code,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=512
)

inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():

    outputs = bert_model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)

    bert_prediction = torch.argmax(probabilities).item()

    bert_confidence = torch.max(probabilities).item() * 100


# Display Results
print("\n")
print("=" * 60)
print("SOURCE CODE VULNERABILITY DETECTION")
print("=" * 60)

print("\nXGBoost")
print("-" * 25)

print(
    f"Prediction : {'Vulnerable' if xgb_prediction else 'Safe'}"
)

print(
    f"Confidence : {xgb_confidence:.2f}%"
)

print("\nCodeBERT")
print("-" * 25)

print(
    f"Prediction : {'Vulnerable' if bert_prediction else 'Safe'}"
)

print(
    f"Confidence : {bert_confidence:.2f}%"
)

print("\n" + "=" * 60)


# Final Recommendation
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

# Security Advice
print("\nRecommendation")
print("-" * 60)

if final == "HIGH RISK":

    print("✓ Vulnerabilities detected by both models.")
    print("✓ Review unsafe memory operations.")
    print("✓ Validate user inputs.")
    print("✓ Avoid strcpy(), gets(), sprintf().")
    print("✓ Use strncpy(), fgets(), snprintf().")

elif final == "SAFE":

    print("✓ No vulnerability detected.")
    print("✓ Continue secure coding practices.")

else:

    print("✓ The two models disagree.")
    print("✓ Manual code review is recommended.")