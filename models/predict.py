import joblib
import pandas as pd
import os

def predict_crop_yield(temperature, precipitation, soil_pH, sunlight, crop_variety, pest_disease):
    # Load the trained model
    model_file_path = os.path.join(os.path.dirname(__file__), '', 'crop_production_model.pkl')
    model = joblib.load(model_file_path)
    
    # Prepare input data as a DataFrame
    input_data = pd.DataFrame({
        'Temperature': [temperature],
        'Precipitation': [precipitation],
        'Soil_pH': [soil_pH],
        'Sunlight': [sunlight],
        'Crop_Variety': [crop_variety],
        'Pest_Disease': [pest_disease]
    })
    
    # Make a prediction
    predicted_yield = model.predict(input_data)
    
    return predicted_yield[0]

if __name__ == '__main__':
    # Example usage
    temperature = 25
    precipitation = 50
    soil_pH = 6.5
    sunlight = 8
    crop_variety = 'Wheat'
    pest_disease = 'No'
    
    predicted_yield = predict_crop_yield(temperature, precipitation, soil_pH, sunlight, crop_variety, pest_disease)
    print(f"Predicted Crop Yield: {predicted_yield} kgs/acre")
