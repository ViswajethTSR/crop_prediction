# app/routes.py
from flask import render_template, request, jsonify
from app import app
from models.predict import predict_crop_yield

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        temperature = float(request.form['temperature'])
        predicted_yield = predict_crop_yield(temperature)
        return jsonify({'predicted_yield': predicted_yield})
    except ValueError:
        return jsonify({'error': 'Invalid input'})

