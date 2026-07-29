from flask import Flask, request, jsonify
import numpy as np
import joblib, os

API_KEY = os.getenv("API_KEY", "dev")
app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "outputs", "models", "model.pkl")
data = joblib.load(MODEL_PATH)
model = data["model"]
scaler = data["scaler"]
feats = data["feature_names"]

@app.route("/")
def index():
    return jsonify({"service": "Cementing Quality Prediction", "features": feats})

@app.route("/predict", methods=["POST"])
def predict():
    key = request.headers.get("X-API-Key") or request.args.get("api_key")
    if key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    body = request.get_json()
    if not body:
        return jsonify({"error": "No input"}), 400
    X = np.array([[body.get(f, 0) for f in feats]])
    if scaler:
        X = scaler.transform(X)
    pred = model.predict(X)[0]
    return jsonify({"prediction": float(pred) if hasattr(pred, '__float__') else str(pred)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
