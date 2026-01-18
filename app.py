from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

# Conversion rate: 1 USD = 300 LKR
USD_TO_LKR = 300

app = Flask(__name__)

# Load the trained model, encoder, and feature names
model_path = 'models/camera_price_model.pkl'
encoder_path = 'models/label_encoder.pkl'
features_path = 'models/feature_names.pkl'

# Check if models exist
if not os.path.exists(model_path):
    print("Model not found! Please run train_model.py first.")
    model = None
else:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    with open(features_path, 'rb') as f:
        feature_names = pickle.load(f)
    print("Model loaded successfully!")

# Brand mapping for display
BRANDS = ['Canon', 'Fujifilm', 'Nikon', 'Olympus', 'Panasonic', 'Sony']
VIDEO_RESOLUTIONS = ['1080p', '4K']


@app.route('/')
def home():
    """Render the home page with the prediction form."""
    return render_template('index.html', brands=BRANDS, video_resolutions=VIDEO_RESOLUTIONS)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        if model is None:
            return render_template('result.html',
                                   error="Model not loaded. Please train the model first by running train_model.py")

        # Get form data
        brand = request.form.get('brand')
        megapixels = float(request.form.get('megapixels'))
        zoom_optical = float(request.form.get('zoom_optical'))
        zoom_digital = float(request.form.get('zoom_digital'))
        screen_size = float(request.form.get('screen_size'))
        weight_grams = float(request.form.get('weight_grams'))
        video_resolution = request.form.get('video_resolution')
        wifi = int(request.form.get('wifi', 0))
        bluetooth = int(request.form.get('bluetooth', 0))

        # Encode brand
        brand_encoded = label_encoder.transform([brand])[0]

        # Encode video resolution
        video_resolution_encoded = 0 if video_resolution == '1080p' else 1

        # Create feature array
        features = np.array([[brand_encoded, megapixels, zoom_optical, zoom_digital,
                            screen_size, weight_grams, video_resolution_encoded, wifi, bluetooth]])

        # Make prediction
        predicted_price = model.predict(features)[0]

        # Convert USD to LKR
        predicted_price_lkr = predicted_price * USD_TO_LKR

        # Prepare camera details for display
        camera_details = {
            'brand': brand,
            'megapixels': megapixels,
            'zoom_optical': zoom_optical,
            'zoom_digital': zoom_digital,
            'screen_size': screen_size,
            'weight_grams': weight_grams,
            'video_resolution': video_resolution,
            'wifi': 'Yes' if wifi else 'No',
            'bluetooth': 'Yes' if bluetooth else 'No'
        }

        return render_template('result.html',
                               prediction=round(predicted_price_lkr, 2),
                               prediction_usd=round(predicted_price, 2),
                               camera_details=camera_details)

    except Exception as e:
        return render_template('result.html',
                               error=f"An error occurred: {str(e)}")


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions (JSON response)."""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        data = request.get_json()

        # Encode brand
        brand_encoded = label_encoder.transform([data['brand']])[0]

        # Encode video resolution
        video_resolution_encoded = 0 if data['video_resolution'] == '1080p' else 1

        # Create feature array
        features = np.array([[
            brand_encoded,
            data['megapixels'],
            data['zoom_optical'],
            data['zoom_digital'],
            data['screen_size'],
            data['weight_grams'],
            video_resolution_encoded,
            data.get('wifi', 0),
            data.get('bluetooth', 0)
        ]])

        # Make prediction
        predicted_price = model.predict(features)[0]

        # Convert USD to LKR
        predicted_price_lkr = predicted_price * USD_TO_LKR

        return jsonify({
            'predicted_price': round(predicted_price_lkr, 2),
            'currency': 'LKR'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/about')
def about():
    """About page with model information."""
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
