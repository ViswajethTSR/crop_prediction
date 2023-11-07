# model/train_model.py
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load the dataset
data = pd.read_csv('../data/crop_data.csv')

# Feature selection and target variable
X = data.drop('Crop_Yield', axis=1)
y = data['Crop_Yield']

# Define which features are categorical and numerical
categorical_features = ['Crop_Variety', 'Pest_Disease']
numerical_features = ['Temperature', 'Precipitation', 'Soil_pH', 'Sunlight']

# Create transformers for preprocessing
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(sparse=False))
])

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Combine the transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the model (Random Forest Regressor)
model = Pipeline(steps=[('preprocessor', preprocessor),
                       ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

# Fit the model
model.fit(X, y)

# Save the trained model
joblib.dump(model, '../models/crop_production_model.pkl')
