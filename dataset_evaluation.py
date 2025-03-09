from dataset_splitting import X_test, y_test
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import pandas as pd 

# Auto-detect latest trained model
model_files = sorted([f for f in os.listdir() if f.startswith("rf_model_v") and f.endswith(".pkl")])
if not model_files:
    raise FileNotFoundError("❌ No trained Random Forest model found!")

latest_model = model_files[-1]
print(f"📌 Loading latest model: {latest_model}")

# Load the trained model
rf_model = joblib.load(latest_model)

# Make predictions on test data
y_pred = rf_model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2%}")

# Show detailed classification report
print("\n📌 Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Display confusion matrix in a labeled DataFrame
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=["Actual Legit", "Actual Phishing"], columns=["Predicted Legit", "Predicted Phishing"])
print("\n📌 Confusion Matrix:")
print(cm_df)
