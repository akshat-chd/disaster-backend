import joblib
import pandas as pd
import requests
from flask import Flask, request, jsonify, make_response
from datetime import datetime, timedelta
import os
import ssl
import certifi

# --- Import extensions ---
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, 
    logout_user, login_required, current_user
)
from flask_bcrypt import Bcrypt
from geopy.geocoders import Nominatim

# --- 1. Initialize App and Config ---
app = Flask(__name__)

# --- Manual CORS Handling (Fixes 403 Errors) ---
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

app.config['SECRET_KEY'] = 'a_very_secret_key_change_this_later' 
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# --- SSL Fix for Geopy (Crucial for Mac) ---
ctx = ssl.create_default_context(cafile=certifi.where())
geolocator = Nominatim(user_agent="disaster-iq-app", ssl_context=ctx)

@login_manager.unauthorized_handler
def unauthorized():
    return jsonify({"error": "Not authorized, please log in."}), 401

# --- User Model ---
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- 2. Load Models ---
try:
    rainfall_model = joblib.load('rainfall_model_v2.pkl')
    print("✅ Rainfall Model loaded.")
except:
    print("🛑 Error: Rainfall Model not found.")
    rainfall_model = None

try:
    # We load it to show it works, but we use live API data for city forecasts
    cyclone_model = joblib.load('cyclone_model_xgb.pkl')
    print("✅ Cyclone Model (XGBoost) loaded.")
except:
    print("⚠️ Cyclone Model not found (Using rule-based logic only).")

# --- 3. Weather Fetching Function ---
def fetch_forecast_data(lat, lon):
    API_URL = "https://api.open-meteo.com/v1/forecast"
    # Added wind_speed_10m_max to the request
    params = { 
        "latitude": lat, 
        "longitude": lon, 
        "daily": [
            "temperature_2m_max", 
            "temperature_2m_min", 
            "apparent_temperature_mean", 
            "sunshine_duration", 
            "precipitation_hours",
            "wind_speed_10m_max" # <-- Needed for Cyclone Risk
        ], 
        "forecast_days": 1, 
        "timezone": "auto" 
    }
    
    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data['daily'])
        
        # Extract wind speed separate from the DataFrame used for Rainfall prediction
        wind_speed = df['wind_speed_10m_max'].iloc[0]

        # Prepare DataFrame for Rainfall Model
        tomorrow_date = datetime.now() + timedelta(days=1)
        df['latitude'] = lat
        df['longitude'] = lon
        df['month'] = tomorrow_date.month
        df['day_of_year'] = tomorrow_date.timetuple().tm_yday
        df['year'] = tomorrow_date.year
        
        # Drop columns not used by the Rainfall model
        df_model = df.drop(['time', 'wind_speed_10m_max'], axis=1) 
        
        # Reorder columns to match training
        model_columns = ['temperature_2m_max', 'temperature_2m_min', 'apparent_temperature_mean', 'sunshine_duration', 'precipitation_hours', 'latitude', 'longitude', 'month', 'day_of_year', 'year']
        df_model = df_model[model_columns]
        
        return df_model, wind_speed
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None, 0

# --- 4. Risk Logic ---
def get_flood_alert(rainfall_mm):
    rainfall_mm = round(rainfall_mm, 2)
    if rainfall_mm < 15.6: return {"risk_level": "Low", "alert_message": f"No significant flood risk ({rainfall_mm} mm)."}
    elif rainfall_mm <= 64.4: return {"risk_level": "Low", "alert_message": f"Low flood risk ({rainfall_mm} mm). Watch for puddles."}
    elif rainfall_mm <= 115.5: return {"risk_level": "Moderate", "alert_message": f"Flood Alert: Heavy rain ({rainfall_mm} mm). Localized flooding likely."}
    elif rainfall_mm <= 204.4: return {"risk_level": "High", "alert_message": f"Flood Warning: Very heavy rain ({rainfall_mm} mm). Widespread flooding likely."}
    else: return {"risk_level": "Severe", "alert_message": f"DANGER: Extreme rain ({rainfall_mm} mm). Severe flooding imminent."}

def get_cyclone_alert(wind_speed):
    wind_speed = round(wind_speed, 2)
    # Wind Speed Scales (Approximate)
    if wind_speed < 31: return {"risk_level": "Low", "alert_message": f"Calm to breezy winds ({wind_speed} km/h). No cyclone risk."}
    elif wind_speed < 50: return {"risk_level": "Moderate", "alert_message": f"Strong winds ({wind_speed} km/h). Be cautious."}
    elif wind_speed < 61: return {"risk_level": "High", "alert_message": f"Depression warning. Very strong winds ({wind_speed} km/h)."}
    elif wind_speed < 88: return {"risk_level": "Severe", "alert_message": f"CYCLONE ALERT: Cyclonic storm winds ({wind_speed} km/h) detected."}
    else: return {"risk_level": "Critical", "alert_message": f"SEVERE CYCLONE WARNING: Dangerous winds ({wind_speed} km/h). Seek shelter!"}

# --- 5. Routes ---
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    try: username, password = data['username'], data['password']
    except: return jsonify({"error": "Missing data"}), 400
    if User.query.filter_by(username=username).first(): return jsonify({"error": "Username exists"}), 409
    hashed = bcrypt.generate_password_hash(password).decode('utf-8')
    user = User(username=username, password=hashed)
    db.session.add(user)
    db.session.commit()
    # Don't log in automatically, let frontend redirect to login
    return jsonify({"message": "User created"}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    user = User.query.filter_by(username=data.get('username')).first()
    if user and bcrypt.check_password_hash(user.password, data.get('password')):
        login_user(user)
        return jsonify({"id": user.id, "username": user.username}), 200
    return jsonify({"error": "Invalid credentials"}), 401

@app.route('/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify({"message": "Logged out"}), 200

@app.route('/@me')
@login_required
def get_current_user():
    return jsonify({"id": current_user.id, "username": current_user.username})

# --- Helper to process predictions ---
def process_prediction(lat, lon, location_name, found_name):
    forecast_df, wind_speed = fetch_forecast_data(lat, lon)
    if forecast_df is None: return jsonify({"error": "Weather fetch failed"}), 500
    
    # Rainfall Prediction (ML)
    pred_rain = float(rainfall_model.predict(forecast_df)[0])
    if pred_rain < 0: pred_rain = 0.0
    flood_alert = get_flood_alert(pred_rain)
    
    # Cyclone Prediction (Rule-based for City Forecast)
    cyclone_alert = get_cyclone_alert(wind_speed)
    
    return jsonify({
        "user": current_user.username,
        "searched_location": location_name,
        "found_location": found_name,
        "latitude": lat,
        "longitude": lon,
        
        # Data keys matching your Frontend
        "predicted_rainfall": round(pred_rain, 2),
        "flood_risk": flood_alert['risk_level'],
        "flood_message": flood_alert['alert_message'],
        
        "wind_speed": round(wind_speed, 2),
        "cyclone_risk": cyclone_alert['risk_level'],
        "cyclone_message": cyclone_alert['alert_message']
    })

# Route 1: Predict by Lat/Lon (GPS)
@app.route('/predict', methods=['POST'])
@login_required 
def predict_geo():
    d = request.get_json()
    return process_prediction(d['latitude'], d['longitude'], "GPS Location", "Your Coordinates")

# Route 2: Predict by Name (Search)
@app.route('/predict_by_name', methods=['POST'])
@login_required
def predict_name():
    name = request.get_json()['location_name']
    # Uses the SSL-fixed geolocator
    loc = geolocator.geocode(name, country_codes="IN")
    if not loc: return jsonify({"error": "Location not found in India"}), 404
    return process_prediction(loc.latitude, loc.longitude, name, loc.address)

if __name__ == '__main__':
    with app.app_context(): db.create_all()
    print("Starting Flask server on port 5001...")
    app.run(debug=True, port=5001)