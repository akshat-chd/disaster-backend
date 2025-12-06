import joblib
import pandas as pd
import requests
import os
import ssl
import certifi
from flask import Flask, request, jsonify
from datetime import datetime, timedelta

# --- Import extensions ---
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, 
    logout_user, login_required, current_user
)
from flask_bcrypt import Bcrypt
from geopy.geocoders import Nominatim
from flask_cors import CORS  # <--- CRITICAL FIX 1

# --- 1. Initialize App and Config ---
app = Flask(__name__)
FRONTEND_URL = "https://disasteriq-frontend.onrender.com"

# --- CRITICAL FIX 2: Better CORS Handling ---
# This replaces your manual 'after_request' code. 
# It handles the "OPTIONS" preflight and Credentials automatically.
CORS(app, origins=[FRONTEND_URL], supports_credentials=True)

# --- Session Config ---
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'a_very_secret_key_change_this_later') 
app.config['SESSION_COOKIE_SAMESITE'] = 'None'
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['REMEMBER_COOKIE_SAMESITE'] = 'None'
app.config['REMEMBER_COOKIE_SECURE'] = True

# Database Config
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# --- SSL Fix for Geopy ---
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

# --- 3. Weather Fetching Function ---
def fetch_forecast_data(lat, lon):
    API_URL = "https://api.open-meteo.com/v1/forecast"
    params = { 
        "latitude": lat, 
        "longitude": lon, 
        "daily": [
            "temperature_2m_max", "temperature_2m_min", "apparent_temperature_mean", 
            "sunshine_duration", "precipitation_hours", "wind_speed_10m_max"
        ], 
        "forecast_days": 1, 
        "timezone": "auto" 
    }
    
    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data['daily'])
        
        wind_speed = df['wind_speed_10m_max'].iloc[0]
        tomorrow_date = datetime.now() + timedelta(days=1)
        df['latitude'] = lat
        df['longitude'] = lon
        df['month'] = tomorrow_date.month
        df['day_of_year'] = tomorrow_date.timetuple().tm_yday
        df['year'] = tomorrow_date.year
        
        df_model = df.drop(['time', 'wind_speed_10m_max'], axis=1) 
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
    if wind_speed < 31: return {"risk_level": "Low", "alert_message": f"Calm to breezy ({wind_speed} km/h)."}
    elif wind_speed < 50: return {"risk_level": "Moderate", "alert_message": f"Strong winds ({wind_speed} km/h)."}
    elif wind_speed < 61: return {"risk_level": "High", "alert_message": f"Depression warning ({wind_speed} km/h)."}
    elif wind_speed < 88: return {"risk_level": "Severe", "alert_message": f"CYCLONE ALERT ({wind_speed} km/h)."}
    else: return {"risk_level": "Critical", "alert_message": f"SEVERE CYCLONE WARNING ({wind_speed} km/h)."}

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
    return jsonify({"message": "User created"}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    user = User.query.filter_by(username=data.get('username')).first()
    if user and bcrypt.check_password_hash(user.password, data.get('password')):
        login_user(user, remember=True)
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

@app.route("/healthz")
def health_check():
    return "OK", 200

# --- Helper to process predictions ---
def process_prediction(lat, lon, location_name, found_name):
    
    # --- CRITICAL FIX 3: Safe User Checking ---
    # This prevents the "AnonymousUserMixin has no attribute username" crash
    if current_user.is_authenticated:
        username = current_user.username
    else:
        # If no cookie found, we don't crash, we just say "Guest"
        username = "Guest"

    forecast_df, wind_speed = fetch_forecast_data(lat, lon)
    if forecast_df is None: return jsonify({"error": "Weather fetch failed"}), 500
    
    pred_rain = float(rainfall_model.predict(forecast_df)[0]) if rainfall_model else 0.0
    if pred_rain < 0: pred_rain = 0.0
    flood_alert = get_flood_alert(pred_rain)
    cyclone_alert = get_cyclone_alert(wind_speed)
    
    return jsonify({
        "user": username, # <--- Uses the safe variable
        "searched_location": location_name,
        "found_location": found_name,
        "latitude": lat,
        "longitude": lon,
        "predicted_rainfall": round(pred_rain, 2),
        "flood_risk": flood_alert['risk_level'],
        "flood_message": flood_alert['alert_message'],
        "wind_speed": round(wind_speed, 2),
        "cyclone_risk": cyclone_alert['risk_level'],
        "cyclone_message": cyclone_alert['alert_message']
    })

# Route 1: Predict by Lat/Lon (GPS)
@app.route('/predict', methods=['POST'])
def predict_geo():
    d = request.get_json()
    return process_prediction(d['latitude'], d['longitude'], "GPS Location", "Your Coordinates")

# Route 2: Predict by Name (Search)
@app.route('/predict_by_name', methods=['POST'])
def predict_name():
    name = request.get_json()['location_name']
    loc = geolocator.geocode(name, country_codes="IN")
    if not loc: return jsonify({"error": "Location not found in India"}), 404
    return process_prediction(loc.latitude, loc.longitude, name, loc.address)

if __name__ == '__main__':
    with app.app_context(): db.create_all()
    print("Starting Flask server on port 5001...")
    app.run(debug=True, port=5001)