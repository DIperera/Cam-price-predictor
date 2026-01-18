import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import os

# Conversion rate: 1 USD = 300 LKR
USD_TO_LKR = 300

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('data/cameras.csv')

print(f"Dataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nStatistical summary:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Encode categorical variable (brand)
label_encoder = LabelEncoder()
df['brand_encoded'] = label_encoder.fit_transform(df['brand'])

# Features and target
X = df[['brand_encoded', 'megapixels', 'zoom_optical', 'zoom_digital',
        'screen_size', 'weight_grams', 'wifi', 'bluetooth']]
y = df['price']

# Add video_resolution encoding
video_resolution_mapping = {'1080p': 0, '4K': 1}
df['video_resolution_encoded'] = df['video_resolution'].map(
    video_resolution_mapping)
X = df[['brand_encoded', 'megapixels', 'zoom_optical', 'zoom_digital',
        'screen_size', 'weight_grams', 'video_resolution_encoded', 'wifi', 'bluetooth']]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Train the model
print("\nTraining XGBoost model...")
model = XGBRegressor(
    n_estimators=100, random_state=42, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8)
model.fit(X_train, y_train)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate the model
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)

print("\n" + "="*50)
print("MODEL EVALUATION RESULTS")
print("="*50)
print(f"\nTraining Set Metrics:")
print(f"  RMSE: Rs {train_rmse * USD_TO_LKR:.2f}")
print(f"  MAE:  Rs {train_mae * USD_TO_LKR:.2f}")
print(f"  R² Score: {train_r2:.4f}")

print(f"\nTest Set Metrics:")
print(f"  RMSE: Rs {test_rmse * USD_TO_LKR:.2f}")
print(f"  MAE:  Rs {test_mae * USD_TO_LKR:.2f}")
print(f"  R² Score: {test_r2:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Save the model and encoder
print("\nSaving model and encoder...")
with open('models/camera_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Save feature names for reference
with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

print("\nModel training complete!")
print("Model saved to: models/camera_price_model.pkl")
print("Label encoder saved to: models/label_encoder.pkl")
print("Feature names saved to: models/feature_names.pkl")

# Test prediction with a sample
print("\n" + "="*50)
print("SAMPLE PREDICTION")
print("="*50)
sample_camera = X_test.iloc[0:1]
predicted_price = model.predict(sample_camera)[0]
actual_price = y_test.iloc[0]
print(f"\nSample camera features:")
print(sample_camera)
print(f"\nPredicted price: Rs {predicted_price * USD_TO_LKR:.2f}")
print(f"Actual price: Rs {actual_price * USD_TO_LKR:.2f}")
print(f"Difference: Rs {abs(predicted_price - actual_price) * USD_TO_LKR:.2f}")
