from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow frontend JS (localhost:3000, etc.)

# Load trained model
model = joblib.load("crop_model.pkl")
print("✅ Model loaded successfully!")

@app.route("/")
def home():
    return "🌱 Crop Prediction API is running locally!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        sample = [[
            data['N'], data['P'], data['K'],
            data['temperature'], data['humidity'],
            data['ph'], data['rainfall']
        ]]
        probs = model.predict_proba(sample)
        top3_idx = np.argsort(probs[0])[-3:][::-1]
        top3_crops = [model.classes_[i] for i in top3_idx]
        top3_probs = [float(probs[0][i]) for i in top3_idx]

        return jsonify({
            top3_crops[0]:top3_probs[0],
            top3_crops[1]:top3_probs[1],
            top3_crops[2]:top3_probs[2]
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)