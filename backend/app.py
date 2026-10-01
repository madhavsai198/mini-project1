import os
import json
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__, static_folder=base_dir, static_url_path='')
CORS(app)

model_path = os.path.join(base_dir, 'model', 'heart_disease_model.json')
scaler_path = os.path.join(base_dir, 'model', 'scaler.json')
metrics_path = os.path.join(base_dir, 'model', 'metrics.json')

model_data = None
scaler_data = None
metrics = None

def load_artifacts():
    global model_data, scaler_data, metrics
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        with open(scaler_path, 'r') as f:
            scaler_data = json.load(f)
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

def predict_proba_pure(input_vector):
    # Scale input
    mean = np.array(scaler_data["mean"])
    std = np.array(scaler_data["std"])
    scaled_vector = (np.array(input_vector) - mean) / std

    weights = np.array(model_data["weights"])
    bias = float(model_data["bias"])

    z = np.dot(scaled_vector, weights) + bias
    z = np.clip(z, -30.0, 30.0)
    proba = 1.0 / (1.0 + np.exp(-z))
    return float(proba)

load_artifacts()

FEATURE_NAMES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

@app.route('/')
def serve_index():
    return send_from_directory(base_dir, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    target = os.path.join(base_dir, path)
    if os.path.exists(target) and not os.path.isdir(target):
        return send_from_directory(base_dir, path)
    return send_from_directory(base_dir, 'index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model_data is not None,
        "scaler_loaded": scaler_data is not None,
        "version": "1.0.0",
        "name": "CardioCare AI Clinical Engine"
    })

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    if metrics is None:
        load_artifacts()
    if metrics:
        return jsonify(metrics)
    return jsonify({"error": "Metrics not found. Run model/train.py first."}), 404

@app.route('/api/presets', methods=['GET'])
def get_presets():
    presets = [
        {
            "id": "low_risk",
            "name": "Low Risk Athlete",
            "description": "34 y/o female with optimal blood pressure & cholesterol",
            "data": {
                "age": 34, "sex": 0, "cp": 2, "trestbps": 112, "chol": 178,
                "fbs": 0, "restecg": 0, "thalach": 168, "exang": 0,
                "oldpeak": 0.2, "slope": 0, "ca": 0, "thal": 1
            }
        },
        {
            "id": "moderate_risk",
            "name": "Moderate Risk Patient",
            "description": "54 y/o male with elevated BP & atypical chest pain",
            "data": {
                "age": 54, "sex": 1, "cp": 1, "trestbps": 138, "chol": 242,
                "fbs": 0, "restecg": 1, "thalach": 142, "exang": 0,
                "oldpeak": 1.4, "slope": 1, "ca": 1, "thal": 2
            }
        },
        {
            "id": "high_risk",
            "name": "High Risk Patient",
            "description": "66 y/o male with exercise angina & high ST depression",
            "data": {
                "age": 66, "sex": 1, "cp": 0, "trestbps": 164, "chol": 310,
                "fbs": 1, "restecg": 1, "thalach": 108, "exang": 1,
                "oldpeak": 2.9, "slope": 1, "ca": 2, "thal": 3
            }
        }
    ]
    return jsonify(presets)

@app.route('/api/predict', methods=['POST'])
def predict():
    global model_data, scaler_data
    if model_data is None or scaler_data is None:
        load_artifacts()
    
    if model_data is None or scaler_data is None:
        return jsonify({"error": "Model missing. Please execute model/train.py"}), 500

    try:
        data = request.json
        input_vector = []
        for feature in FEATURE_NAMES:
            if feature not in data:
                return jsonify({"error": f"Missing required parameter: {feature}"}), 400
            input_vector.append(float(data[feature]))

        proba = predict_proba_pure(input_vector)
        risk_percent = round(proba * 100, 1)

        if risk_percent < 30.0:
            tier = "Low Risk"
            badge_color = "emerald"
            recommendations = [
                "Maintain current active lifestyle and routine annual health screening.",
                "Sustain balanced diet low in saturated fats.",
                "Target resting blood pressure below 120/80 mm Hg."
            ]
        elif risk_percent < 65.0:
            tier = "Moderate Risk"
            badge_color = "amber"
            recommendations = [
                "Schedule a formal preventative cardiology assessment.",
                "Adopt low-sodium Mediterranean diet rich in antioxidants.",
                "Engage in 150+ minutes of moderate aerobic activity weekly."
            ]
        else:
            tier = "High Risk"
            badge_color = "rose"
            recommendations = [
                "Immediate consultation recommended with a specialist cardiologist.",
                "Complete full clinical diagnostic ECG and treadmill stress testing.",
                "Closely monitor lipid profile, blood pressure, and ST parameters."
            ]

        feature_importances = metrics.get('feature_importances', {}) if metrics else {}
        contributions = []
        for i, feat in enumerate(FEATURE_NAMES):
            val = input_vector[i]
            imp = feature_importances.get(feat, 0.08)
            contributions.append({
                "feature": feat,
                "value": val,
                "importance": round(imp * 100, 2)
            })

        contributions.sort(key=lambda x: x['importance'], reverse=True)

        return jsonify({
            "success": True,
            "risk_percentage": risk_percent,
            "risk_tier": tier,
            "badge_color": badge_color,
            "recommendations": recommendations,
            "top_drivers": contributions[:4]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"[*] Starting CardioCare AI Server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=True)
