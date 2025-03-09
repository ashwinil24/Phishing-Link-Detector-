import joblib

try:
    model = joblib.load("rf_model_v4.pkl")
    print("✅ Model loaded successfully!")
    print(f"📌 Model Classes: {model.classes_}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
