import os
import pandas as pd
import numpy as np

def build_features():
    print("Loading analysis-ready dataset...")
    df = pd.read_csv("data/processed/analysis_ready_data.csv")
    df['date'] = pd.to_datetime(df['date'])
    
    # Ensure chronological order per SKU
    df = df.sort_values(by=['sku_id', 'date']).reset_index(drop=True)
    
    print("Generating time-series lag and rolling features...")
    
    # 1. Lag Features (Historical daily demand)
    lags = [1, 7, 14, 28]
    for lag in lags:
        df[f'units_sold_lag_{lag}'] = df.groupby('sku_id')['units_sold'].shift(lag)
        
    # 2. Rolling Window Features (Moving averages and std)
    windows = [7, 14, 28]
    for w in windows:
        # Shift by 1 first so we don't include today's actual sales in prediction input
        df[f'units_sold_roll_mean_{w}'] = (
            df.groupby('sku_id')['units_sold']
            .transform(lambda x: x.shift(1).rolling(w).mean())
        )
        df[f'units_sold_roll_std_{w}'] = (
            df.groupby('sku_id')['units_sold']
            .transform(lambda x: x.shift(1).rolling(w).std())
        )
        
    # 3. Pricing & Margin Features
    df['price_discount_ratio'] = np.where(
        df['Selling_Price'] > 0, 
        df['unit_price'] / df['Selling_Price'], 
        1.0
    )
    
    # Drop initial rows with NaNs caused by 28-day lag calculation
    df_clean = df.dropna(subset=['units_sold_lag_28', 'units_sold_roll_mean_28']).copy()
    
    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/featured_data.csv"
    df_clean.to_csv(output_path, index=False)
    
    print(f"Feature engineering complete! Saved to: {output_path}")
    print(f"Dataset Shape: {df_clean.shape}")

if __name__ == "__main__":
    build_features()