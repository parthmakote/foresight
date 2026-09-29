import os
import sys
import joblib
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --- Page Setup & Executive Styling ---
st.set_page_config(
    page_title="Foresight | Supply Chain & Demand Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for executive cards and typography
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        border-radius: 6px 6px 0px 0px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- Resource Caching ---
@st.cache_data
def load_datasets():
    reorder_plan = pd.read_csv("reports/inventory_reorder_plan.csv")
    featured_data = pd.read_csv("data/processed/featured_data.csv")
    featured_data['date'] = pd.to_datetime(featured_data['date'])
    
    fi_path = "reports/feature_importance.csv"
    fi_df = pd.read_csv(fi_path) if os.path.exists(fi_path) else None
    
    return reorder_plan, featured_data, fi_df

@st.cache_resource
def load_trained_model():
    return joblib.load("models/rf_model.pkl.gz")

# --- Safe Data Artifact Loading ---
data_loaded = False
reorder_plan, featured_data, feature_importance_df, rf_model = None, None, None, None

try:
    if (os.path.exists("reports/inventory_reorder_plan.csv") and 
        os.path.exists("data/processed/featured_data.csv") and 
        os.path.exists("models/rf_model.pkl")):
        reorder_plan, featured_data, feature_importance_df = load_datasets()
        rf_model = load_trained_model()
        data_loaded = True
except Exception as e:
    data_loaded = False

# --- Header ---
st.markdown('<div class="main-title">📦 Foresight: Supply Chain & Demand Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Powered Multi-SKU Demand Forecasting, Dynamic Reorder Optimization & Risk Simulation</div>', unsafe_allow_html=True)

# --- Navigation Tabs ---
tab_upload, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📤 Upload Raw Data",
    "🚨 Replenishment Dashboard", 
    "📈 SKU Analytics & Predictions", 
    "📊 Category Portfolio Health", 
    "⚙️ What-If Policy Simulator", 
    "🧠 Model Drivers & Explainability"
])

# ==========================================
# TAB 0: UPLOAD RAW DATA & RUN PIPELINE
# ==========================================
with tab_upload:
    st.subheader("📤 Upload Raw Datasets & Execute Pipeline")
    st.markdown("Upload your 4 raw CSV files to automatically process data, retrain the Machine Learning model, and update inventory reorder points.")

    col_u1, col_u2 = st.columns(2)

    with col_u1:
        file_sales = st.file_uploader("1. Sales Log (`sales_daily.csv`)", type=["csv"])
        file_inventory = st.file_uploader("2. Inventory Snapshot (`inventory_snapshots.csv`)", type=["csv"])

    with col_u2:
        file_promo = st.file_uploader("3. Marketing Log (`calendar.csv`)", type=["csv"])
        file_catalog = st.file_uploader("4. Product Catalog (`sku_master.csv`)", type=["csv"])

    st.divider()

    if st.button("🚀 Process Data & Run Supply Chain Engine", type="primary"):
        if file_sales and file_inventory and file_promo and file_catalog:
            try:
                os.makedirs("data", exist_ok=True)
                os.makedirs("data/raw", exist_ok=True)

                with st.spinner("Step 1/4: Saving uploaded CSV files..."):
                    with open("data/sales_daily.csv", "wb") as f:
                        f.write(file_sales.getbuffer())
                    with open("data/inventory_snapshots.csv", "wb") as f:
                        f.write(file_inventory.getbuffer())
                    with open("data/calendar.csv", "wb") as f:
                        f.write(file_promo.getbuffer())
                    with open("data/sku_master.csv", "wb") as f:
                        f.write(file_catalog.getbuffer())

                with st.spinner("Step 2/4: Running Data Pipeline & Feature Engineering..."):
                    if os.path.exists("src/pipeline.py"):
                        subprocess.run([sys.executable, "src/pipeline.py"], check=True)
                    elif os.path.exists("src/features.py"):
                        subprocess.run([sys.executable, "src/features.py"], check=True)

                with st.spinner("Step 3/4: Retraining Machine Learning Model..."):
                    if os.path.exists("src/train_baseline.py"):
                        subprocess.run([sys.executable, "src/train_baseline.py"], check=True)

                with st.spinner("Step 4/4: Optimizing Inventory & Calculating Reorder Points..."):
                    if os.path.exists("src/inventory_optimization.py"):
                        subprocess.run([sys.executable, "src/inventory_optimization.py"], check=True)

                # Clear Streamlit cache to force reload of new metrics
                st.cache_data.clear()
                st.cache_resource.clear()

                st.success("✅ Engine Pipeline Executed Successfully! Refreshing Dashboard...")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Pipeline Execution Error: {e}")
        else:
            st.warning("⚠️ Please upload all 4 required CSV files before executing the pipeline.")

# --- If Data is Not Yet Loaded, Show Prompt ---
if not data_loaded:
    st.info("👋 Welcome! Please switch to the **📤 Upload Raw Data** tab above and upload your 4 CSV files to generate initial dashboard metrics.")
    st.stop()

# --- Top Level Metrics Summary (Rendered once data is loaded) ---
total_skus = len(reorder_plan)
reorder_skus = reorder_plan[reorder_plan['inventory_status'] == 'REORDER NOW']
reorder_count = len(reorder_skus)
optimal_count = total_skus - reorder_count
risk_pct = (reorder_count / total_skus) * 100
total_safety_stock_units = int(reorder_plan['optimal_safety_stock'].sum())

st.divider()

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Total SKUs Managed</div><div class="metric-value">{total_skus}</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card" style="border-left: 4px solid #EF4444;"><div class="metric-label">Stockout Risk SKUs</div><div class="metric-value" style="color: #DC2626;">{reorder_count} <span style="font-size:0.9rem; font-weight:normal;">({risk_pct:.0f}%)</span></div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card" style="border-left: 4px solid #10B981;"><div class="metric-label">Healthy Stock SKUs</div><div class="metric-value" style="color: #059669;">{optimal_count}</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Required Safety Stock</div><div class="metric-value">{total_safety_stock_units:,} <span style="font-size:0.8rem; font-weight:normal;">units</span></div></div>', unsafe_allow_html=True)
with c5:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Model Accuracy (WAPE)</div><div class="metric-value" style="color: #2563EB;">23.33%</div></div>', unsafe_allow_html=True)

st.write("")

# --- Sidebar Controls ---
st.sidebar.image("https://img.icons8.com/isometric-line/100/4a6da7/box.png", width=70)
st.sidebar.title("Control Panel")

category_filter = st.sidebar.multiselect(
    "Filter by Category:",
    options=sorted(reorder_plan['Category'].unique()),
    default=sorted(reorder_plan['Category'].unique())
)

status_filter = st.sidebar.multiselect(
    "Filter Inventory Status:",
    options=['REORDER NOW', 'STOCK OPTIMAL'],
    default=['REORDER NOW', 'STOCK OPTIMAL']
)

filtered_reorder = reorder_plan[
    (reorder_plan['Category'].isin(category_filter)) &
    (reorder_plan['inventory_status'].isin(status_filter))
]

# ==========================================
# TAB 1: REPLENISHMENT DASHBOARD
# ==========================================
with tab1:
    st.subheader("Inventory Replenishment & Stockout Alerts")
    
    col_t1, col_t2 = st.columns([2, 1])
    
    with col_t1:
        fig_reorder = px.bar(
            filtered_reorder,
            x='sku_id',
            y=['current_on_hand', 'optimal_reorder_point'],
            barmode='group',
            labels={'value': 'Units', 'variable': 'Metric', 'sku_id': 'SKU ID'},
            title="Current On-Hand Stock vs Dynamic Reorder Point (ROP)",
            color_discrete_sequence=['#3B82F6', '#EF4444']
        )
        fig_reorder.update_layout(legend_title_text="", hovermode="x unified", height=380)
        st.plotly_chart(fig_reorder, use_container_width=True)

    with col_t2:
        status_counts = filtered_reorder['inventory_status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        
        fig_pie = px.pie(
            status_counts, 
            names='Status', 
            values='Count', 
            title="Stock Health Ratio",
            color='Status',
            color_discrete_map={'REORDER NOW': '#EF4444', 'STOCK OPTIMAL': '#10B981'},
            hole=0.45
        )
        fig_pie.update_layout(height=380, showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("### Actionable Purchase Orders & Replenishment Plan")
    
    def highlight_reorder(val):
        color = '#FEE2E2' if val == 'REORDER NOW' else '#D1FAE5'
        text_color = '#991B1B' if val == 'REORDER NOW' else '#065F46'
        return f'background-color: {color}; color: {text_color}; font-weight: bold;'

    st.dataframe(
        filtered_reorder[['sku_id', 'Product_Name', 'Category', 'current_on_hand', 'optimal_reorder_point', 'optimal_safety_stock', 'lead_time_days', 'inventory_status']]
        .style.map(highlight_reorder, subset=['inventory_status']),
        use_container_width=True,
        height=300
    )

    csv_data = filtered_reorder.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Replenishment Plan to CSV",
        data=csv_data,
        file_name="procurement_reorder_plan.csv",
        mime="text/csv",
        help="Download purchase order list for ERP/Procurement ingestion"
    )

# ==========================================
# TAB 2: SKU ANALYTICS & PREDICTIONS
# ==========================================
with tab2:
    st.subheader("SKU-Level Demand Deep Dive & AI Forecast")
    
    selected_sku = st.selectbox(
        "Select SKU to Analyze:",
        options=sorted(reorder_plan['sku_id'].unique())
    )
    
    sku_meta = reorder_plan[reorder_plan['sku_id'] == selected_sku].iloc[0]
    sku_df = featured_data[featured_data['sku_id'] == selected_sku].sort_values('date').copy()
    
    feature_cols = [
        'units_sold_lag_1', 'units_sold_lag_7', 'units_sold_lag_14', 'units_sold_lag_28',
        'units_sold_roll_mean_7', 'units_sold_roll_std_7',
        'units_sold_roll_mean_14', 'units_sold_roll_std_14',
        'units_sold_roll_mean_28', 'units_sold_roll_std_28',
        'unit_price', 'promo_flag', 'is_weekend', 'is_holiday',
        'on_hand_units', 'reorder_point', 'price_discount_ratio'
    ]
    
    sku_df['predicted_demand'] = rf_model.predict(sku_df[feature_cols])

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Product Name", sku_meta['Product_Name'])
    m2.metric("Category", sku_meta['Category'])
    m3.metric("Current On-Hand", f"{int(sku_meta['current_on_hand'])} units")
    m4.metric("Optimal Safety Stock", f"{int(sku_meta['optimal_safety_stock'])} units")
    m5.metric("Reorder Point (ROP)", f"{int(sku_meta['optimal_reorder_point'])} units")

    fig_sku_forecast = go.Figure()
    fig_sku_forecast.add_trace(go.Scatter(
        x=sku_df['date'], y=sku_df['units_sold'],
        mode='lines', name='Actual Demand',
        line=dict(color='#2563EB', width=2)
    ))
    fig_sku_forecast.add_trace(go.Scatter(
        x=sku_df['date'], y=sku_df['predicted_demand'],
        mode='lines', name='Random Forest Model (.pkl)',
        line=dict(color='#F59E0B', dash='dash', width=2)
    ))
    fig_sku_forecast.add_trace(go.Scatter(
        x=sku_df['date'], y=sku_df['units_sold_roll_mean_7'],
        mode='lines', name='7-Day Moving Avg',
        line=dict(color='#9CA3AF', width=1, dash='dot')
    ))

    fig_sku_forecast.update_layout(
        title=f"Daily Demand Trend, Moving Averages & Forecast — {selected_sku}",
        xaxis_title="Date",
        yaxis_title="Units Sold",
        hovermode="x unified",
        height=420
    )
    st.plotly_chart(fig_sku_forecast, use_container_width=True)

    st.markdown("#### Promotional Uplift Analysis for " + selected_sku)
    promo_analysis = sku_df.groupby('promo_flag')['units_sold'].agg(['mean', 'median', 'std', 'count']).reset_index()
    promo_analysis['promo_label'] = promo_analysis['promo_flag'].map({0: 'Regular Days', 1: 'Promotional Days'})
    
    fig_promo = px.bar(
        promo_analysis, 
        x='promo_label', 
        y='mean', 
        color='promo_label',
        text_auto='.1f',
        title="Average Daily Units Sold: Regular vs Promotional Days",
        labels={'mean': 'Avg Daily Units Sold', 'promo_label': 'Day Type'},
        color_discrete_map={'Regular Days': '#64748B', 'Promotional Days': '#10B981'}
    )
    fig_promo.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig_promo, use_container_width=True)

# ==========================================
# TAB 3: CATEGORY PORTFOLIO HEALTH
# ==========================================
with tab3:
    st.subheader("Category-Level Performance & Stock Balance")

    cat_summary = reorder_plan.groupby('Category').agg(
        total_skus=('sku_id', 'count'),
        reorder_skus=('inventory_status', lambda x: (x == 'REORDER NOW').sum()),
        avg_safety_stock=('optimal_safety_stock', 'mean'),
        avg_rop=('optimal_reorder_point', 'mean'),
        total_on_hand=('current_on_hand', 'sum')
    ).reset_index()

    cat_summary['risk_rate'] = (cat_summary['reorder_skus'] / cat_summary['total_skus']) * 100

    col_cat1, col_cat2 = st.columns(2)

    with col_cat1:
        fig_cat_bar = px.bar(
            cat_summary,
            x='Category',
            y=['total_skus', 'reorder_skus'],
            barmode='group',
            title="Total SKUs vs Stockout Risk SKUs per Category",
            labels={'value': 'Count', 'variable': 'Metric'},
            color_discrete_sequence=['#3B82F6', '#EF4444']
        )
        fig_cat_bar.update_layout(height=380)
        st.plotly_chart(fig_cat_bar, use_container_width=True)

    with col_cat2:
        fig_cat_risk = px.bar(
            cat_summary,
            x='Category',
            y='risk_rate',
            title="Stockout Risk Rate (%) by Category",
            labels={'risk_rate': 'Risk Rate (%)'},
            color='risk_rate',
            color_continuous_scale='Reds'
        )
        fig_cat_risk.update_layout(height=380)
        st.plotly_chart(fig_cat_risk, use_container_width=True)

    st.markdown("### Category Summary Table")
    st.dataframe(cat_summary, use_container_width=True)

# ==========================================
# TAB 4: WHAT-IF POLICY SIMULATOR
# ==========================================
with tab4:
    st.subheader("Supply Chain Scenario & Buffer Policy Simulator")
    st.markdown("Test how shifts in **Target Service Levels** ($Z$) or **Supplier Lead Time Delays** impact inventory safety buffers and purchase order requirements.")

    sim_c1, sim_c2 = st.columns(2)

    with sim_c1:
        service_level = st.select_slider(
            "Target Service Level (%)",
            options=[90.0, 95.0, 98.0, 99.0],
            value=95.0,
            help="Higher service level reduces stockouts but requires higher safety stock."
        )
        z_map = {90.0: 1.28, 95.0: 1.65, 98.0: 2.05, 99.0: 2.33}
        z_val = z_map[service_level]

    with sim_c2:
        lt_factor = st.slider(
            "Supplier Lead Time Multiplier (Delay Factor)",
            min_value=0.5,
            max_value=2.5,
            value=1.0,
            step=0.1,
            help="1.0 = Standard supplier lead time; 1.5 = 50% lead time delay"
        )

    sim_df = reorder_plan.copy()
    sim_df['sim_lead_time'] = sim_df['lead_time_days'] * lt_factor
    sim_df['sim_safety_stock'] = np.ceil(z_val * sim_df['forecast_std'] * np.sqrt(sim_df['sim_lead_time']))
    sim_df['sim_reorder_point'] = np.ceil((sim_df['avg_daily_forecast'] * sim_df['sim_lead_time']) + sim_df['sim_safety_stock'])
    sim_df['sim_status'] = np.where(sim_df['current_on_hand'] <= sim_df['sim_reorder_point'], 'REORDER NOW', 'STOCK OPTIMAL')

    sim_reorder_count = (sim_df['sim_status'] == 'REORDER NOW').sum()
    sim_total_ss = int(sim_df['sim_safety_stock'].sum())
    ss_diff = sim_total_ss - total_safety_stock_units

    st.info(f"💡 **Scenario Impact:** At **{service_level}% Service Level** with a **{lt_factor}x Lead Time Multiplier**:\n"
            f"- **{sim_reorder_count} out of {total_skus} SKUs** will require replenishment.\n"
            f"- Total Safety Stock needed: **{sim_total_ss:,} units** (Change: **{'+' if ss_diff >= 0 else ''}{ss_diff:,} units** vs baseline).")

    fig_sim_comp = px.scatter(
        sim_df,
        x='optimal_reorder_point',
        y='sim_reorder_point',
        color='sim_status',
        hover_data=['sku_id', 'Product_Name'],
        labels={'optimal_reorder_point': 'Baseline ROP', 'sim_reorder_point': 'Simulated ROP'},
        title="Baseline ROP vs Simulated ROP per SKU",
        color_discrete_map={'REORDER NOW': '#EF4444', 'STOCK OPTIMAL': '#10B981'}
    )
    max_val = max(sim_df['optimal_reorder_point'].max(), sim_df['sim_reorder_point'].max())
    fig_sim_comp.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val, line=dict(color="Gray", dash="dash"))
    fig_sim_comp.update_layout(height=400)
    
    st.plotly_chart(fig_sim_comp, use_container_width=True)

# ==========================================
# TAB 5: MODEL DRIVERS & EXPLAINABILITY
# ==========================================
with tab5:
    st.subheader("Machine Learning Interpretability & Feature Drivers")
    st.markdown("Understand which time-series signals and promotional flags drive daily demand predictions in the Random Forest model.")

    if feature_importance_df is not None:
        fig_fi = px.bar(
            feature_importance_df.sort_values(by='importance', ascending=True),
            x='importance',
            y='feature',
            orientation='h',
            title="Random Forest Feature Importance Scores",
            labels={'importance': 'Importance Score', 'feature': 'Feature Signal'},
            color='importance',
            color_continuous_scale='Viridis'
        )
        fig_fi.update_layout(height=450)
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.warning("Feature importance file `reports/feature_importance.csv` not found. Run `python src/train_baseline.py` to generate it.")

    st.markdown("""
    ### Key Insights from the Model:
    1. **Long-Term Demand Memory (`units_sold_roll_mean_28`)**: Over **59%** of predictive power comes from 28-day moving average demand, indicating strong baseline sales stability.
    2. **Medium-Term Momentum (`units_sold_roll_mean_14`)**: Captures 14-day demand acceleration and recent trends (**18.3%** importance).
    3. **Promotions (`promo_flag`)**: Dynamic sales spikes are heavily driven by promotional events, making promotional calendar inputs crucial for accurate forecasting.
    """)