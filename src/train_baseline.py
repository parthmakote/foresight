import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import joblib 

def evaluate_forecast():
    print("Loading featured dataset...")
    df = pd.read_csv("data/processed/featured_data.csv")
    df['date'] = pd.to_datetime(df['date'])
    
    # 1. Chronological Train-Test Split (Cutoff: Oct 1, 2025)
    split_date = pd.to_datetime("2025-10-01")
    train = df[df['date'] < split_date].copy()
    test = df[df['date'] >= split_date].copy()
    
    print(f"Train Set: {train.shape[0]} rows ({train['date'].min().date()} to {train['date'].max().date()})")
    print(f"Test Set:  {test.shape[0]} rows ({test['date'].min().date()} to {test['date'].max().date()})")
    
    # Features selection
    feature_cols = [
        'units_sold_lag_1', 'units_sold_lag_7', 'units_sold_lag_14', 'units_sold_lag_28',
        'units_sold_roll_mean_7', 'units_sold_roll_std_7',
        'units_sold_roll_mean_14', 'units_sold_roll_std_14',
        'units_sold_roll_mean_28', 'units_sold_roll_std_28',
        'unit_price', 'promo_flag', 'is_weekend', 'is_holiday',
        'on_hand_units', 'reorder_point', 'price_discount_ratio'
    ]
    target_col = 'units_sold'
    
    X_train, y_train = train[feature_cols], train[target_col]
    X_test, y_test = test[feature_cols], test[target_col]
    
    # -------------------------------------------------------------
    # Model 1: Naïve 7-Day Moving Average Baseline
    # -------------------------------------------------------------
    naive_preds = test['units_sold_roll_mean_7']
    naive_mae = mean_absolute_error(y_test, naive_preds)
    naive_rmse = root_mean_squared_error(y_test, naive_preds)
    
    # -------------------------------------------------------------
    # Model 2: Random Forest Regressor
    # -------------------------------------------------------------
    print("\nTraining Random Forest Regressor...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    model_path = "models/rf_model.pkl"
    
    # Save as .pkl with level 3 compression (matches inventory_optimization.py)
    joblib.dump(rf_model, model_path, compress=3)
    print(f"Model saved successfully to: {model_path}")
    
    rf_preds = rf_model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_rmse = root_mean_squared_error(y_test, rf_preds)
    
    # Calculate WAPE (Weighted Absolute Percentage Error)
    wape_naive = (np.sum(np.abs(y_test - naive_preds)) / np.sum(y_test)) * 100
    wape_rf = (np.sum(np.abs(y_test - rf_preds)) / np.sum(y_test)) * 100
    
    print("\n" + "="*45)
    print("      DEMAND FORECASTING MODEL RESULTS      ")
    print("="*45)
    print(f"1. Naïve 7-Day MA Baseline:")
    print(f"   - MAE:  {naive_mae:.2f} units")
    print(f"   - RMSE: {naive_rmse:.2f} units")
    print(f"   - WAPE: {wape_naive:.2f}%")
    
    print(f"\n2. Random Forest Regressor:")
    print(f"   - MAE:  {rf_mae:.2f} units")
    print(f"   - RMSE: {rf_rmse:.2f} units")
    print(f"   - WAPE: {wape_rf:.2f}%")
    print("="*45)
    
    # Feature Importance
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    os.makedirs("reports", exist_ok=True)
    importances.to_csv("reports/feature_importance.csv", index=False)
    print("\nTop 5 Most Important Features:")
    print(importances.head(5).to_string(index=False))

if __name__ == "__main__":
    evaluate_forecast()