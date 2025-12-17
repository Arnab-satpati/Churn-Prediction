from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# --- 1. Load the Full Pipeline ---
# This contains: Preprocessor (Scaling/Encoding) + Random Forest Model
MODEL_PATH = 'models/rf_pipeline.pkl'
model = joblib.load(MODEL_PATH)

@app.route('/')
def home():
    return "Churn Prediction API is Running. Use the /predict endpoint."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()
        
        # Convert JSON to DataFrame (model expects a DF with named columns)
        input_df = pd.DataFrame([data])
        
        # Make prediction
        # The pipeline automatically handles scaling and encoding
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        return jsonify({
            'churn_prediction': int(prediction),
            'probability': round(float(probability), 4),
            'status': 'Success'
        })

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'Fail'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)