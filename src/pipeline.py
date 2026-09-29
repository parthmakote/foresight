import os
import pandas as pd

def load_data():
    """Load raw datasets from data directory."""
    print("Loading raw CSV files...")
    sales = pd.read_csv("data/sales_daily.csv")
    sku = pd.read_csv("data/sku_master.csv")
    calendar = pd.read_csv("data/calendar.csv")
    inventory = pd.read_csv("data/inventory_snapshots.csv")
    return sales, sku, calendar, inventory

def clean_and_merge(sales, sku, calendar, inventory):
    """Clean, standardize, and join all tables into a unified dataset."""
    print("Standardizing column schema...")
    
    # 1. Standardize column names
    sales = sales.rename(columns={
        'Date': 'date',
        'SKU': 'sku_id',
        'Units_Sold': 'units_sold',
        'Revenue': 'revenue',
        'Price': 'unit_price',
        'Promotion': 'promo_flag'
    })
    
    sku = sku.rename(columns={'SKU': 'sku_id'})
    
    calendar = calendar.rename(columns={'date': 'date'})
    
    inventory = inventory.rename(columns={
        'Snapshot_Date': 'date',
        'SKU': 'sku_id',
        'Current_Stock': 'on_hand_units',
        'On_Order': 'on_order_units',
        'Lead_Time_Days': 'lead_time_days',
        'Safety_Stock': 'safety_stock',
        'Reorder_Point': 'reorder_point',
        'Inventory_Value': 'inventory_value'
    })

    # 2. Convert date strings to datetime objects
    sales['date'] = pd.to_datetime(sales['date'])
    calendar['date'] = pd.to_datetime(calendar['date'])
    inventory['date'] = pd.to_datetime(inventory['date'])

    # 3. Deduplicate sales records
    sales = sales.drop_duplicates()

    # 4. Merge datasets
    print("Merging tables...")
    # Join SKU metadata onto sales
    df = pd.merge(sales, sku, on='sku_id', how='left')
    
    # Join Calendar features onto sales
    df = pd.merge(df, calendar, on='date', how='left')
    
    # Sort prior to inventory snapshot propagation
    df = df.sort_values(by=['sku_id', 'date']).reset_index(drop=True)
    
    # Join Inventory snapshots
    df = pd.merge(df, inventory, on=['date', 'sku_id'], how='left')

    # 5. Fill missing values
    # Forward-fill and backward-fill periodic snapshot metrics for daily rows
    inv_cols = [
        'on_hand_units', 'on_order_units', 'lead_time_days', 
        'safety_stock', 'reorder_point', 'inventory_value'
    ]
    df[inv_cols] = df.groupby('sku_id')[inv_cols].ffill().bfill()

    # Impute missing categorical values
    df['holiday'] = df['holiday'].fillna('None')
    df['promotion_event'] = df['promotion_event'].fillna('None')
    df['promo_flag'] = df['promo_flag'].fillna(0).astype(int)

    return df

def run_pipeline():
    sales, sku, calendar, inventory = load_data()
    merged_df = clean_and_merge(sales, sku, calendar, inventory)

    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/analysis_ready_data.csv"
    
    merged_df.to_csv(output_path, index=False)
    
    print("\n--- Pipeline Run Summary ---")
    print(f"Output Path: {output_path}")
    print(f"Total Records Processed: {merged_df.shape[0]}")
    print(f"Total Feature Columns: {merged_df.shape[1]}")
    print(f"Remaining Missing Values: {merged_df.isnull().sum().sum()}")

if __name__ == "__main__":
    run_pipeline()