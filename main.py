import joblib
import pandas as pd
import requests
from flask import Flask, request, jsonify, make_response
from datetime import datetime, timedelta
import os

# --- Import extensions ---
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, 
    logout_user, login_required, current_user
)
from flask_bcrypt import Bcrypt

# --- Import Geopy ---
from geopy.geocoders import Nominatim

# --- 1. Initialize App and Config ---
app = Flask(__name__)

# --- Manual CORS Handling ---
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        res = make_response()
        res.headers.set('Access-Control-Allow-Origin', 'http://localhost:5173')
        res.headers.set('Access-Control-Allow-Headers', 'Content-Type')
        res.headers.set('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        res.headers.set('Access-Control-Allow-Credentials', 'true')
        return res

@app.after_request
def after_request(response):
    response.headers.set('Access-Control-Allow-Origin', 'http://localhost:5173')
    response.headers.set('Access-Control-Allow-Credentials', 'true')
    return response

# --- Configuration ---
app.config['SECRET_KEY'] = 'a_very_secret_key_change_this_later' 
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# --- Initialize Extensions ---
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# --- Initialize Geocoder ---
geolocator = Nominatim(user_agent="disaster-iq-app")

# --- API-friendly unauthorized handler ---
@login_manager.unauthorized_handler
def unauthorized():
    return jsonify({"error": "Not authorized, please log in."}), 401

# --- Define User Model ---
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# --- User Loader ---
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# --- 2. Load the ML Model ---
try:
    model = joblib.load('rainfall_model_v2.pkl')
    print("✅ Model loaded successfully from 'rainfall_model_v2.pkl'")
except FileNotFoundError:
    print("🛑 Error: 'rainfall_model_v2.pkl' not found.")
    model = None
except Exception as e:
    print(f"🛑 Error loading model: {e}")
    model = None


# --- 3. Weather Fetching Function ---
def fetch_forecast_data(lat, lon):
    print(f"Fetching forecast for: ({lat}, {lon})")
    API_URL = "https://api.open-meteo.com/v1/forecast"
    params = { "latitude": lat, "longitude": lon, "daily": ["temperature_2m_max", "temperature_2m_min", "apparent_temperature_mean", "sunshine_duration", "precipitation_hours"], "forecast_days": 1, "timezone": "auto" }
    
    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data['daily'])
        tomorrow_date = datetime.now() + timedelta(days=1)
        df['latitude'] = lat
        df['longitude'] = lon
        df['month'] = tomorrow_date.month
        df['day_of_year'] = tomorrow_date.timetuple().tm_yday
        df['year'] = tomorrow_date.year
        df = df.drop('time', axis=1)
        model_columns = [ 'temperature_2m_max', 'temperature_2m_min', 'apparent_temperature_mean', 'sunshine_duration', 'precipitation_hours', 'latitude', 'longitude', 'month', 'day_of_year', 'year' ]
        df = df[model_columns]
        print("✅ Forecast data processed successfully.")
        return df
    except Exception as e:
        print(f"🛑 Error processing forecast data: {e}")
        return None

# --- 4. Rainfall Alert Function ---
def get_flood_alert(rainfall_mm):
    rainfall_mm = round(rainfall_mm, 2)
    if rainfall_mm < 15.6:
        return {"risk_level": "Low", "alert_message": f"No significant flood risk. Light to moderate rain ({rainfall_mm} mm) expected."}
    elif rainfall_mm <= 64.4:
        return {"risk_level": "Low", "alert_message": f"Low flood risk. Moderate rain ({rainfall_mm} mm) expected. Be cautious of waterlogging."}
    elif rainfall_mm <= 115.5:
        return {"risk_level": "Moderate", "alert_message": f"Flood Alert: Heavy rainfall ({rainfall_mm} mm) predicted. Risk of localized flooding."}
    elif rainfall_mm <= 204.4:
        return {"risk_level": "High", "alert_message": f"Flood Warning: Very heavy rainfall ({rainfall_mm} mm) predicted. High risk of widespread flooding."}
    else:
        return {"risk_level": "Severe", "alert_message": f"DANGER: Extremely heavy rainfall ({rainfall_mm} mm) predicted. Severe, widespread flooding is imminent."}

# --- 5. Authentication Endpoints ---
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    try: username, password = data['username'], data['password']
    except KeyError: return jsonify({"error": "Please provide 'username' and 'password'"}), 400
    if User.query.filter_by(username=username).first(): return jsonify({"error": "Username already exists"}), 409
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    new_user = User(username=username, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()
    login_user(new_user)
    return jsonify({"id": new_user.id, "username": new_user.username}), 200

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    try: username, password = data['username'], data['password']
    except KeyError: return jsonify({"error": "Please provide 'username' and 'password'"}), 400
    user = User.query.filter_by(username=username).first()
    if user and bcrypt.check_password_hash(user.password, password):
        login_user(user) 
        return jsonify({"id": user.id, "username": user.username}), 200
    else: return jsonify({"error": "Invalid username or password"}), 401

@app.route('/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify({"message": "You have been logged out."}), 200

@app.route('/@me')
@login_required
def get_current_user():
    return jsonify({"id": current_user.id, "username": current_user.username})

# --- 6. Prediction Endpoints ---

# Route 1: Predict by Lat/Lon (for the map page or "Use My Location")
@app.route('/predict', methods=['POST'])
@login_required 
def predict_rainfall():
    if model is None: return jsonify({"error": "Model is not loaded."}), 500
    try:
        data = request.get_json()
        lat, lon = data['latitude'], data['longitude']
    except Exception as e: return jsonify({"error": f"Invalid input: {e}"}), 400
    
    forecast_df = fetch_forecast_data(lat, lon)
    if forecast_df is None: return jsonify({"error": "Could not retrieve weather data."}), 500
        
    try:
        prediction = model.predict(forecast_df)
        predicted_rainfall = float(prediction[0])
        if predicted_rainfall < 0: predicted_rainfall = 0.0
        alert_info = get_flood_alert(predicted_rainfall)
        
        # --- Cyclone check ---
        cyclone_message = ""
        try:
            location_details = geolocator.reverse((lat, lon), language='en')
            # Simple heuristic: if 'water' isn't mentioned in type, assume less risk
            if location_details and 'water' not in location_details.raw.get('type', ''):
                 cyclone_message = "This location is inland; severe cyclone risk is low."
        except Exception:
            pass # Ignore geocoding errors for this extra check

        print(f"Prediction for user '{current_user.username}': {predicted_rainfall:.2f} mm")
        return jsonify({ 
            "user": current_user.username, 
            "latitude": lat, 
            "longitude": lon, 
            "predicted_rainfall_tomorrow_mm": round(predicted_rainfall, 2), 
            "cyclone_risk_message": cyclone_message,
            **alert_info 
        })
    except Exception as e: return jsonify({"error": f"Prediction error: {e}"}), 500

# Route 2: Predict by Name (This is the one giving you 404!)
@app.route('/predict_by_name', methods=['POST'])
@login_required
def predict_by_name():
    if model is None: return jsonify({"error": "Model is not loaded."}), 500
    
    try:
        data = request.get_json()
        location_name = data['location_name']
    except Exception as e:
        return jsonify({"error": f"Invalid input. Must provide 'location_name'. Error: {e}"}), 400

    try:
        # 1. Geocode the location name
        location = geolocator.geocode(location_name, country_codes="IN")
        
        if not location:
            return jsonify({"error": f"Location '{location_name}' not found in India."}), 404
            
        lat, lon = location.latitude, location.longitude
        found_name = location.address
        
        # 2. Run prediction logic
        forecast_df = fetch_forecast_data(lat, lon)
        if forecast_df is None: return jsonify({"error": "Could not retrieve weather data."}), 500
        
        prediction = model.predict(forecast_df)
        predicted_rainfall = float(prediction[0])
        if predicted_rainfall < 0: predicted_rainfall = 0.0
        alert_info = get_flood_alert(predicted_rainfall)
        
        # 3. Cyclone check
        cyclone_message = ""
        try:
            if 'water' not in location.raw.get('type', ''):
                 cyclone_message = "This location is inland; severe cyclone risk is low."
        except Exception:
            pass

        print(f"Prediction by name for '{location_name}': {predicted_rainfall:.2f} mm")
        
        return jsonify({
            "user": current_user.username,
            "searched_location": location_name,
            "found_location": found_name,
            "latitude": lat,
            "longitude": lon,
            "predicted_rainfall_tomorrow_mm": round(predicted_rainfall, 2),
            "cyclone_risk_message": cyclone_message,
            **alert_info
        })
        
    except Exception as e:
        print(f"🛑 Error during geocoding/prediction: {e}")
        return jsonify({"error": f"An error occurred: {e}"}), 500

# --- 7. Run the API Server ---
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    print("Starting Flask server...")
    app.run(debug=True, port=5001)