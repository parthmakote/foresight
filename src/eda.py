import pandas as pd
import os

def run_eda():
    print("Loading analysis-ready dataset...")
    df = pd.read_csv("data/processed/analysis_ready_data.csv")
    
    print("\n" + "="*40)
    print("        BUSINESS METRICS OVERVIEW        ")
    print("="*40)
    
    # 1. Financial Performance
    total_revenue = df['revenue'].sum()
    total_units = df['units_sold'].sum()
    avg_order_price = df['unit_price'].mean()
    
    print(f"Total Revenue Generated:  ${total_revenue:,.2f}")
    print(f"Total Units Sold:         {total_units:,}")
    print(f"Average Unit Selling Price: ${avg_order_price:,.2f}")
    
    # 2. Performance by Category
    print("\n--- Sales by Category ---")
    cat_summary = df.groupby('Category').agg(
        Total_Revenue=('revenue', 'sum'),
        Total_Units=('units_sold', 'sum')
    ).reset_index().sort_values(by='Total_Revenue', ascending=False)
    print(cat_summary.to_string(index=False))
    
    # 3. Top 5 Performing SKUs
    print("\n--- Top 5 Best-Selling SKUs (Revenue) ---")
    top_skus = df.groupby(['sku_id', 'Product_Name', 'Category'])['revenue'].sum().reset_index()
    top_skus = top_skus.sort_values(by='revenue', ascending=False).head(5)
    print(top_skus.to_string(index=False))
    
    # 4. Inventory Risk Assessment (Stockout Risks)
    stockouts = df[df['on_hand_units'] < df['reorder_point']]
    stockout_count = len(stockouts)
    print("\n--- Inventory Risk Analysis ---")
    print(f"Days where On-Hand Inventory fell below Reorder Point: {stockout_count:,} instances")
    
    # Save key metrics summary to file
    os.makedirs("reports", exist_ok=True)
    cat_summary.to_csv("reports/category_summary.csv", index=False)
    top_skus.to_csv("reports/top_skus.csv", index=False)
    print("\nEDA Summary reports exported to `reports/` folder.")

if __name__ == "__main__":
    run_eda()