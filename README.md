# 📷 Camera Price Predictor

A machine learning web application that predicts camera prices based on technical specifications using **XGBoost**. Built with Flask for easy deployment and interactive predictions.

## 🚀 Features

- **XGBoost Regression Model** - High-performance gradient boosting for accurate price predictions
- **Interactive Web Interface** - User-friendly Flask application with modern UI
- **9 Feature Inputs** - Brand, megapixels, zoom, screen size, weight, video quality, connectivity
- **Real-time Predictions** - Instant price estimates in LKR (Sri Lankan Rupees)
- **Model Persistence** - Pre-trained models saved for quick deployment
- **Data Preprocessing** - Label encoding for categorical features

## 📊 Dataset

The model is trained on a dataset of 100 camera records with the following features:

| Feature          | Description             | Range/Values                                     |
| ---------------- | ----------------------- | ------------------------------------------------ |
| Brand            | Camera manufacturer     | Canon, Nikon, Sony, Fujifilm, Panasonic, Olympus |
| Megapixels       | Camera resolution       | 10-65 MP                                         |
| Optical Zoom     | Optical zoom capability | 0-30x                                            |
| Digital Zoom     | Digital zoom capability | 0-10x                                            |
| Screen Size      | LCD screen size         | 2.5-3.5 inches                                   |
| Weight           | Camera weight           | 200-1000 grams                                   |
| Video Resolution | Video recording quality | 1080p / 4K                                       |
| WiFi             | WiFi connectivity       | Yes / No                                         |
| Bluetooth        | Bluetooth connectivity  | Yes / No                                         |

## 🛠️ Tech Stack

- **Python 3.8+**
- **Flask** - Web framework
- **XGBoost** - Machine learning model
- **Scikit-learn** - Data preprocessing and metrics
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations
- **Pickle** - Model serialization

## 📁 Project Structure

```
Cam-price-predictor/
├── data/
│   └── cameras.csv              # Training dataset
├── models/
│   ├── camera_price_model.pkl   # Trained XGBoost model
│   ├── label_encoder.pkl        # Brand encoder
│   └── feature_names.pkl        # Feature list
├── static/
│   └── css/
│       └── style.css            # Styling
├── templates/
│   ├── index.html               # Input form
│   └── result.html              # Prediction results
├── Procedures/
│   └── procedures.txt           # Detailed documentation
├── train_model.py               # Model training script
├── app.py                       # Flask web server
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/Cam-price-predictor.git
   cd Cam-price-predictor
   ```

2. **Create virtual environment**

   ```bash
   python -m venv .venv
   ```

3. **Activate virtual environment**
   - Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

### Step 1: Train the Model

Train the XGBoost model on the camera dataset:

```bash
python train_model.py
```

**Output:**

- Trains XGBoost model on camera specifications
- Saves model files to `models/` directory
- Displays evaluation metrics (RMSE, MAE, R² score)

### Step 2: Run the Web Application

Start the Flask development server:

```bash
python app.py
```

Then open your browser and navigate to: `http://127.0.0.1:5000`

### Step 3: Make Predictions

1. Select camera specifications from the dropdown menus and input fields
2. Click "Predict Price"
3. View the predicted price in Sri Lankan Rupees (LKR)

## 📈 Model Performance

The XGBoost model achieves strong performance on the test set:

- **R² Score**: ~0.88 (88% variance explained)
- **RMSE**: ~₨92,000 average error
- **MAE**: ~₨44,000 mean absolute error

### Feature Importance

The most influential features for price prediction:

1. **Megapixels** (44%) - Most important factor
2. **Video Resolution** (26%) - 4K significantly increases price
3. **Weight** (15%) - Heavier cameras tend to be more expensive
4. **Digital Zoom** (7%)
5. **Optical Zoom** (6%)

## 🔑 Key Files

### `train_model.py`

- Loads and preprocesses camera data
- Encodes categorical features (brand, video resolution)
- Trains XGBoost regression model
- Evaluates performance metrics
- Saves trained models and encoders

### `app.py`

- Flask web server setup
- Loads pre-trained models
- Handles form submissions
- Makes predictions on new data
- Renders HTML templates with results

### `requirements.txt`

Lists all Python dependencies including Flask, XGBoost, scikit-learn, pandas, and numpy.

## 🌐 API Endpoints

| Endpoint   | Method | Description                                |
| ---------- | ------ | ------------------------------------------ |
| `/`        | GET    | Home page with input form                  |
| `/predict` | POST   | Processes form data and returns prediction |

## 💡 Example Prediction

**Input:**

- Brand: Canon
- Megapixels: 24 MP
- Optical Zoom: 10x
- Digital Zoom: 4x
- Screen Size: 3.0 inches
- Weight: 500g
- Video: 4K
- WiFi: Yes
- Bluetooth: Yes

**Output:** Predicted Price: ₨180,000 (approximately)

## 🐛 Troubleshooting

**Model files not found:**

- Ensure you've run `train_model.py` before starting the Flask app

**Import errors:**

- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check that virtual environment is activated

**Port already in use:**

- Change the port in `app.py`: `app.run(debug=True, port=5001)`

## 📚 Documentation

For detailed technical documentation, code explanations, and library usage, see [`Procedures/procedures.txt`](Procedures/procedures.txt).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

Created as a machine learning demonstration project.

---

**Note:** Prices are converted to LKR using a fixed rate (1 USD = 300 LKR). Update the conversion rate in `app.py` as needed.
