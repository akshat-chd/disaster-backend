import requests
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def fetch_weather_data(lat, lon, start_date, end_date):
    """
    Fetches historical weather data from Open-Meteo for a specific location.
    """
    print(f"Fetching data for location: ({lat}, {lon})...")
    
    # Open-Meteo Historical API endpoint
    API_URL = "https://archive-api.open-meteo.com/v1/archive"
    
    # Define the weather variables we want to train on (our features)
    # and our target ('precipitation_sum')
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "precipitation_sum",          # Our target (Y)
            "temperature_2m_max",         # Feature
            "temperature_2m_min",         # Feature
            "apparent_temperature_mean",  # Feature
            "sunshine_duration",          # Feature
            "precipitation_hours"         # Feature
        ],
        "timezone": "auto"
    }
    
    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()  # Raises an error for bad responses (4xx or 5xx)
        data = response.json()
        
        # Convert the daily data into a pandas DataFrame
        df = pd.DataFrame(data['daily'])
        
        # Add latitude and longitude as features
        df['latitude'] = lat
        df['longitude'] = lon
        
        return df
        
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def train_and_save_model():
    """
    Fetches data for multiple locations, processes it,
    trains an XGBoost model, and saves it to a file.
    """
    
    # --- 1. Data Collection ---
    # We train on data from multiple cities to build a robust model
    locations = {
        "New Delhi": (28.61, 77.23),
        "Mumbai": (19.08, 72.88),
        "London": (51.51, -0.13),
        "New York": (40.71, -74.01),
        "Tokyo": (35.69, 139.69)
    }
    
    start_date = "2020-01-01"
    end_date = "2024-12-31" # Use 5 years of data
    
    all_data = []
    
    for city, (lat, lon) in locations.items():
        print(f"\n--- Processing {city} ---")
        city_data = fetch_weather_data(lat, lon, start_date, end_date)
        if city_data is not None:
            all_data.append(city_data)
            
    if not all_data:
        print("No data fetched. Exiting training.")
        return

    # Combine all data into one big DataFrame
    master_df = pd.concat(all_data, ignore_index=True)
    
    # --- 2. Feature Engineering ---
    print("\nProcessing and engineering features...")
    
    # Convert 'time' column to datetime objects
    master_df['time'] = pd.to_datetime(master_df['time'])
    
    # Create new time-based features
    master_df['month'] = master_df['time'].dt.month
    master_df['day_of_year'] = master_df['time'].dt.dayofyear
    master_df['year'] = master_df['time'].dt.year
    
    # Drop the original 'time' column as it's no longer needed
    master_df = master_df.drop('time', axis=1)
    
    # Handle any missing values (e.g., fill with 0 or mean)
    master_df = master_df.fillna(0)
    
    # --- 3. Model Training ---
    print("Defining features (X) and target (y)...")
    
    # Our target 'y' is what we want to predict
    y = master_df['precipitation_sum']
    
    # Our features 'X' are all the other columns
    X = master_df.drop('precipitation_sum', axis=1)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training a new XGBoost rainfall model...")
    # Initialize and train the XGBoost model
    # (These parameters can be tuned for better performance)
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,  # Number of trees
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate the model
    print("Evaluating model...")
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
# Then manually take the square root to get the RMSE

    print(f"Model RMSE (Root Mean Squared Error): {rmse:.4f}")
    
    # --- 4. Save the Model ---
    # We save the new, smarter model as 'v2'
    model_filename = 'rainfall_model_v2.pkl'
    print(f"\n✅ Model trained and saving as '{model_filename}'...")
    joblib.dump(model, model_filename)
    print(f"✅ Model saved successfully.")

# --- Run the script ---
if __name__ == "__main__":
    train_and_save_model()