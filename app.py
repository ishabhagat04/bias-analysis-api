
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import json
import os

# Load model and preprocessor
model = joblib.load('bias_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

# Load model info
with open('model_info.json', 'r') as f:
    model_info = json.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return f"""<!DOCTYPE html>
<html>
<head>
    <title>AI Bias Analysis API</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; }}
        .container {{ max-width: 800px; margin: 0 auto; }}
        .card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        code {{ background: #e9ecef; padding: 2px 6px; border-radius: 4px; }}
        pre {{ background: #2d2d2d; color: #f8f9fa; padding: 15px; border-radius: 8px; overflow-x: auto; }}
        .badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 14px; }}
        .badge-green {{ background: #27ae60; color: white; }}
        .badge-blue {{ background: #3498db; color: white; }}
        hr {{ border: none; border-top: 2px solid #ecf0f1; margin: 30px 0; }}
        a {{ color: #3498db; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>⚖️ AI Bias Analysis - Model API</h1>

        <div class="card">
            <p><strong>Model:</strong> {model_info['model_name']}</p>
            <p><strong>Accuracy:</strong> {model_info['accuracy']:.4f}</p>
            <p><strong>Sensitive Feature:</strong> {model_info['sensitive_feature']}</p>
            <p>
                <span class="badge badge-green">✓ Active</span>
                <span class="badge badge-blue">v1.0</span>
            </p>
        </div>

        <hr>

        <h3>📤 Test the API</h3>
        <p>Send POST request to <code>/predict</code> with JSON data</p>

        <h4>Example 1: Using raw features</h4>
        <pre>{{
  "features": [30, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}}</pre>

        <h4>Example 2: Using named features</h4>
        <pre>{{
  "age": 30,
  "priors_count": 0,
  "race": 1
}}</pre>

        <hr>

        <h3>🔍 Available Endpoints</h3>
        <ul>
            <li><code>GET /</code> - This home page</li>
            <li><code>POST /predict</code> - Make a prediction</li>
            <li><code>GET /fairness</code> - View fairness metrics</li>
            <li><code>GET /health</code> - Health check</li>
        </ul>

        <hr>

        <p style="color: #7f8c8d; font-size: 14px;">
            🔍 <a href="/fairness">View Fairness Metrics</a>
        </p>
    </div>
</body>
</html>"""

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if data is None:
            return jsonify({'error': 'No JSON data provided'}), 400

        # Get features from input
        if 'features' in data:
            features = data['features']
            # Create DataFrame with proper column names
            if isinstance(features, list):
                input_df = pd.DataFrame([features], columns=model_info['features'][:len(features)])
            else:
                input_df = pd.DataFrame([list(features.values())], columns=model_info['features'])
        else:
            # Extract from named fields
            feature_values = []
            for feature in model_info['features']:
                feature_values.append(data.get(feature, 0))
            input_df = pd.DataFrame([feature_values], columns=model_info['features'])

        # Preprocess
        processed = preprocessor.transform(input_df)

        # Predict
        prediction = model.predict(processed)

        # Get probability if available
        probability = None
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(processed)

        # Check fairness warning
        fairness_warning = None
        if model_info['sensitive_feature'] in input_df.columns:
            sensitive_value = input_df.iloc[0][model_info['sensitive_feature']]
            if sensitive_value == 1:
                fairness_warning = f"⚠️ This prediction uses sensitive feature '{model_info['sensitive_feature']}' (value: 1)"

        response = {
            'prediction': int(prediction[0]),
            'prediction_label': 'High Risk' if prediction[0] == 1 else 'Low Risk',
            'probability': probability[0].tolist() if probability is not None else None,
            'fairness_note': fairness_warning,
            'model_accuracy': model_info['accuracy']
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/fairness', methods=['GET'])
def fairness():
    """Endpoint to get model fairness metrics"""
    return jsonify({
        'model_name': model_info['model_name'],
        'accuracy': model_info['accuracy'],
        'spd': model_info.get('spd', 'N/A'),
        'sensitive_feature': model_info['sensitive_feature'],
        'note': 'SPD closer to 0 means fairer model'
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
