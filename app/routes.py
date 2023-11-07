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
        precipitation = float(request.form['precipitation'])
        soil_pH = float(request.form['soil_pH'])
        sunlight = float(request.form['sunlight'])
        crop_variety = request.form['crop_variety']
        pest_disease = request.form['pest_disease']

        predicted_yield = predict_crop_yield(temperature, precipitation, soil_pH, sunlight, crop_variety, pest_disease)
        return jsonify({'predicted_yield': predicted_yield})
    except ValueError:
        return jsonify({'error': 'Invalid input'})
