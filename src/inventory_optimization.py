import os
import joblib
import numpy as np
import pandas as pd

def optimize_inventory():
    print("Loading featured dataset and saved model...")
    df = pd.read_csv("data/processed/featured_data.csv")
    df['date'] = pd.to_datetime(df['date'])
    
    split_date = pd.to_datetime("2025-10-01")
    test = df[df['date'] >= split_date].copy()
    
    feature_cols = [
        'units_sold_lag_1', 'units_sold_lag_7', 'units_sold_lag_14', 'units_sold_lag_28',
        'units_sold_roll_mean_7', 'units_sold_roll_std_7',
        'units_sold_roll_mean_14', 'units_sold_roll_std_14',
        'units_sold_roll_mean_28', 'units_sold_roll_std_28',
        'unit_price', 'promo_flag', 'is_weekend', 'is_holiday',
        'on_hand_units', 'reorder_point', 'price_discount_ratio'
    ]
    
    # Load model from disk instead of retraining
    rf_model = joblib.load("models/rf_model.pkl")
    test['pred_demand'] = rf_model.predict(test[feature_cols])
    
    z_score = 1.65 # 95% Service Level
    
    sku_summary = test.groupby(['sku_id', 'Product_Name', 'Category']).agg(
        avg_daily_forecast=('pred_demand', 'mean'),
        forecast_std=('pred_demand', 'std'),
        lead_time_days=('lead_time_days', 'mean'),
        current_on_hand=('on_hand_units', 'last'),
        legacy_reorder_point=('reorder_point', 'last')
    ).reset_index()
    
    sku_summary['demand_during_lead_time'] = sku_summary['avg_daily_forecast'] * sku_summary['lead_time_days']
    sku_summary['optimal_safety_stock'] = np.ceil(z_score * sku_summary['forecast_std'] * np.sqrt(sku_summary['lead_time_days']))
    sku_summary['optimal_reorder_point'] = np.ceil(sku_summary['demand_during_lead_time'] + sku_summary['optimal_safety_stock'])
    
    sku_summary['inventory_status'] = np.where(
        sku_summary['current_on_hand'] <= sku_summary['optimal_reorder_point'],
        'REORDER NOW',
        'STOCK OPTIMAL'
    )
    
    os.makedirs("reports", exist_ok=True)
    sku_summary.to_csv("reports/inventory_reorder_plan.csv", index=False)
    print("Inventory optimization updated using saved .pkl model!")

if __name__ == "__main__":
    optimize_inventory()