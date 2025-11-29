import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def train_cyclone_model():
    print("Loading processed data...")
    try:
        X = pd.read_csv('X_cyclone.csv')
        y = pd.read_csv('y_cyclone.csv')
    except FileNotFoundError:
        print("Error: .csv files not found. Run prepare_cyclone_data.py first.")
        return

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training XGBoost Cyclone Model...")
    
    # Initialize XGBoost Regressor
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,    # More trees for complex patterns
        learning_rate=0.05,  # Slower learning for better accuracy
        max_depth=6,         # Deeper trees
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)
    
    print(f"\n--- Model Performance ---")
    print(f"RMSE (Wind Speed Error): {rmse:.2f} knots")
    print(f"R-Squared (Accuracy): {r2:.4f}")
    
    # Save Model
    joblib.dump(model, 'cyclone_model_xgb.pkl')
    print("\n✅ Model trained and saved as 'cyclone_model_xgb.pkl'")

if __name__ == "__main__":
    train_cyclone_model()


### Step 3: Run the Scripts

# Now, run them in order in your terminal:


# # 1. Prepare Data
# python prepare_cyclone_data.py

# # 2. Train Model
# python train_cyclone_model.py