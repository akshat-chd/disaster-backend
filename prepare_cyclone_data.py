import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

def prepare_data(filepath='cyclone_data.csv'):
    print("Loading dataset...")
    # Skip second row (units)
    df = pd.read_csv(filepath, skiprows=[1], low_memory=False)
    
    # Keep relevant columns
    cols = ['SID', 'ISO_TIME', 'LAT', 'LON', 'USA_WIND', 'USA_PRES']
    df = df[cols].copy()
    
    # Convert types
    for col in ['LAT', 'LON', 'USA_WIND', 'USA_PRES']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()
    print(f"Cleaned data: {len(df)} records.")

    # --- Feature Engineering for XGBoost ---
    # We want to predict 'Will it be a cyclone?' (Wind > 34 knots)
    # We create 'Lag' features: What was the wind/pressure 6 hours ago?
    
    # Sort by Storm ID and Time
    df = df.sort_values(['SID', 'ISO_TIME'])
    
    # Shift data to get previous time step (t-1)
    df['Prev_LAT'] = df.groupby('SID')['LAT'].shift(1)
    df['Prev_LON'] = df.groupby('SID')['LON'].shift(1)
    df['Prev_WIND'] = df.groupby('SID')['USA_WIND'].shift(1)
    df['Prev_PRES'] = df.groupby('SID')['USA_PRES'].shift(1)
    
    # Drop rows where we don't have previous history (first record of a storm)
    df = df.dropna()
    
    # Define Target: Current Wind Speed
    X = df[['Prev_LAT', 'Prev_LON', 'Prev_WIND', 'Prev_PRES']]
    y = df['USA_WIND']
    
    print(f"Training Data Shape: {X.shape}")
    
    # Save the processed dataframes
    X.to_csv('X_cyclone.csv', index=False)
    y.to_csv('y_cyclone.csv', index=False)
    print("Data processing complete. Saved X_cyclone.csv and y_cyclone.csv")

if __name__ == "__main__":
    prepare_data()