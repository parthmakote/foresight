import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

def plot_results():
    df = pd.read_csv("data/processed/featured_data.csv")
    df['date'] = pd.to_datetime(df['date'])
    
    split_date = pd.to_datetime("2025-10-01")
    train = df[df['date'] < split_date].copy()
    test = df[df['date'] >= split_date].copy()
    
    feature_cols = [
        'units_sold_lag_1', 'units_sold_lag_7', 'units_sold_lag_14', 'units_sold_lag_28',
        'units_sold_roll_mean_7', 'units_sold_roll_std_7',
        'units_sold_roll_mean_14', 'units_sold_roll_std_14',
        'units_sold_roll_mean_28', 'units_sold_roll_std_28',
        'unit_price', 'promo_flag', 'is_weekend', 'is_holiday',
        'on_hand_units', 'reorder_point', 'price_discount_ratio'
    ]
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(train[feature_cols], train['units_sold'])
    test['pred_demand'] = rf.predict(test[feature_cols])
    
    # Aggregate aggregate daily demand across all SKUs for clear trend viewing
    daily_agg = test.groupby('date')[['units_sold', 'pred_demand']].sum().reset_index()
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=daily_agg, x='date', y='units_sold', label='Actual Sales (Units)', color='#1f77b4', linewidth=2)
    sns.lineplot(data=daily_agg, x='date', y='pred_demand', label='Random Forest Forecast', color='#ff7f0e', linestyle='--', linewidth=2)
    
    plt.title('Q4 2025 Demand Forecast vs Actual Sales (Total Portfolio)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Total Units Sold', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    os.makedirs("reports", exist_ok=True)
    chart_path = "reports/forecast_vs_actual_q4.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    
    print(f"Forecast chart generated successfully: {chart_path}")

if __name__ == "__main__":
    plot_results()