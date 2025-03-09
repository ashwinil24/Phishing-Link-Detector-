from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from feature_extraction import extract_features

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow cross-origin requests

# Load the trained model
try:
    rf_model = joblib.load("rf_model_v4.pkl")
    
    expected_columns = list(rf_model.feature_names_in_) if hasattr(rf_model, "feature_names_in_") else [
        "url_length", "dot_count", "slash_count", "subdomain_count", "suspicious_keyword",
        "special_char_count", "digit_count", "url_encoding_count", "has_ip", "is_shortened",
        "repeated_char_count", "https_present", "has_at_symbol"
    ]
except Exception as e:
    print(f"❌ Error loading model: {e}")
    rf_model = None
    expected_columns = []

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.content_type != "application/json":
            return jsonify({"error": "Invalid Content-Type. Expected application/json"}), 400

        data = request.get_json()
        if not data or "url" not in data:
            return jsonify({"error": "No URL provided"}), 400

        url = data["url"].strip()
        if not url:
            return jsonify({"error": "URL cannot be empty"}), 400

        # Extract features
        features = extract_features(url)
        if features is None:
            return jsonify({"error": "Feature extraction failed"}), 500

        features_df = pd.DataFrame([features])

        print(f"📌 Extracted Features: {features_df.columns.tolist()}")
        print(f"📌 Expected Features: {expected_columns}")

        # Ensure model is loaded
        if rf_model is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        # Check if 'has_at_symbol' is included in the model's feature set
        if rf_model:
            model_features = list(rf_model.feature_names_in_) if hasattr(rf_model, "feature_names_in_") else []
            print(f"📌 Model Features: {model_features}")


        # Align features with the model's expected input
        features_df = features_df.reindex(columns=expected_columns, fill_value=0)

        print(f"📌 Final Feature Values: \n{features_df}")

        # Make prediction
        prediction = rf_model.predict(features_df)[0]
        probability = rf_model.predict_proba(features_df)[0]

        # Debugging: Print probability values
        print(f"📌 Raw Probability Output: {probability} (Length: {len(probability)})")

        # Ensure probability array is valid
        if len(probability) == 2:
            legit_prob, fake_prob = round(probability[0], 4), round(probability[1], 4)
        else:
            legit_prob, fake_prob = (1.0, 0.0) if prediction == 0 else (0.0, 1.0)

        result = "Fake" if prediction == 1 else "Legit"

        return jsonify({
            "url": url,
            "result": result,
            "probability": {
                "Legit": legit_prob,
                "Fake": fake_prob
            }
        })

    except Exception as e:
        print(f"❌ Error in /predict: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)