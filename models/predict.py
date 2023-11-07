# model/predict.py
import joblib
import os


def predict_crop_yield(temperature):
    # Load the trained model
    model_file_path = os.path.join(os.path.dirname(__file__), '', 'crop_production_model.pkl')
    model = joblib.load(model_file_path)
    
    # Make a prediction
    predicted_yield = model.predict([[temperature]])
    
    return predicted_yield[0]
