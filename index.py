from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app,origins="*")

# Load trained model
model = joblib.load("crop_model.pkl")
print("✅ Model loaded successfully!")

# Optional: API key for security
API_KEY = os.environ.get("API_KEY", "changeme123")

@app.before_request
def verify_key():
    key = request.headers.get("x-api-key")
    if key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

@app.route("/")
def home():
    return "🌱 Crop Prediction API is live!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        sample = [[
            data['N'],
            data['P'],
            data['K'],
            data['temperature'],
            data['humidity'],
            data['ph'],
            data['rainfall']
        ]]
        probs = model.predict_proba(sample)
        top3_idx = np.argsort(probs[0])[-3:][::-1]
        top3_crops = [model.classes_[i] for i in top3_idx]
        top3_probs = [float(probs[0][i]) for i in top3_idx]

        return jsonify({
            "Top 3 Crops": [
                {"crop": crop, "probability": round(prob*100,2)}
                for crop, prob in zip(top3_crops, top3_probs)
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
