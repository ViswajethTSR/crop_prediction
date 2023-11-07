# model/train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the dataset
data = pd.read_csv('../data/crop_data.csv')

# Feature selection and target variable
X = data[['Temperature']]  # You may have more features in a real-world scenario
y = data['Crop_Yield']

# Train a model (Random Forest Regressor in this example)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the trained model
joblib.dump(model, '../models/crop_production_model.pkl')
