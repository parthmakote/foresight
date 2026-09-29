import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure reports directory exists
os.makedirs("reports/visuals", exist_ok=True)

# Set plotting theme
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10})

def generate_visuals():
    print("Loading processed dataset...")
    df = pd.read_csv("data/processed/analysis_ready_data.csv")
    df['date'] = pd.to_datetime(df['date'])
    
    # -------------------------------------------------------------
    # Chart 1: Monthly Sales Revenue Trend
    # -------------------------------------------------------------
    df['year_month'] = df['date'].dt.to_period('M').astype(str)
    monthly_rev = df.groupby('year_month')['revenue'].sum().reset_index()

    plt.figure(figsize=(12, 5))
    plt.plot(monthly_rev['year_month'], monthly_rev['revenue'] / 1e6, marker='o', color='#1f77b4', linewidth=2.5)
    plt.title('Monthly Sales Revenue Trend (2024 - 2025)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Year-Month', fontsize=11, fontweight='bold')
    plt.ylabel('Revenue ($ Millions)', fontsize=11, fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('reports/visuals/monthly_revenue_trend.png', dpi=300)
    plt.close()
    print("Saved: reports/visuals/monthly_revenue_trend.png")

    # -------------------------------------------------------------
    # Chart 2: Total Revenue by Subcategory & Category
    # -------------------------------------------------------------
    cat_rev = df.groupby(['Category', 'Subcategory'])['revenue'].sum().reset_index().sort_values('revenue', ascending=False)

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=cat_rev, x='revenue', y='Subcategory', hue='Category', dodge=False, palette='Set2')
    plt.title('Total Revenue by Subcategory & Category', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Revenue ($)', fontsize=11, fontweight='bold')
    plt.ylabel('Subcategory', fontsize=11, fontweight='bold')
    ax.xaxis.set_major_formatter('${x:,.0f}')
    plt.legend(title='Category', loc='lower right')
    plt.tight_layout()
    plt.savefig('reports/visuals/revenue_by_category.png', dpi=300)
    plt.close()
    print("Saved: reports/visuals/revenue_by_category.png")

    # -------------------------------------------------------------
    # Chart 3: Impact of Promotions on Average Daily Revenue
    # -------------------------------------------------------------
    promo_summary = df.groupby('promo_flag')['revenue'].mean().reset_index()
    promo_summary['Promotion'] = promo_summary['promo_flag'].map({0: 'No Promo', 1: 'Promo Active'})

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=promo_summary, x='Promotion', y='revenue', palette=['#3498db', '#e74c3c'])
    plt.title('Impact of Promotions on Average Daily Revenue', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Promotion Status', fontsize=11, fontweight='bold')
    plt.ylabel('Average Daily Revenue ($)', fontsize=11, fontweight='bold')
    for p in ax.patches:
        ax.annotate(f'${p.get_height():,.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 8), textcoords='offset points', fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/visuals/promo_impact.png', dpi=300)
    plt.close()
    print("Saved: reports/visuals/promo_impact.png")

    # -------------------------------------------------------------
    # Chart 4: Inventory Stockout Risk Distribution
    # -------------------------------------------------------------
    df['stockout_risk'] = df['on_hand_units'] < df['reorder_point']
    risk_by_cat = df.groupby('Category')['stockout_risk'].mean().reset_index()
    risk_by_cat['stockout_pct'] = risk_by_cat['stockout_risk'] * 100

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=risk_by_cat, x='Category', y='stockout_pct', palette='YlOrRd')
    plt.title('Percentage of Operational Days Below Reorder Point (Stockout Risk)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Category', fontsize=11, fontweight='bold')
    plt.ylabel('% of Operational Days in Risk', fontsize=11, fontweight='bold')
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 8), textcoords='offset points', fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/visuals/stockout_risk_by_category.png', dpi=300)
    plt.close()
    print("Saved: reports/visuals/stockout_risk_by_category.png")

if __name__ == "__main__":
    generate_visuals()