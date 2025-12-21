import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# --- PAGE CONFIG ---
st.set_page_config(page_title="StratSim: Executive Decision Suite", layout="wide")

# --- CUSTOM STYLING ---
st.markdown("""
<style>
    .main { background-color: #fcfcfc; }
    .metric-card {
        background-color: white;
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #ececec;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-label { font-size: 14px; color: #6e6e6e; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 32px; color: #1a1a1a; font-weight: 800; margin-top: 10px; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #0047AB; color: white; }
</style>
""", unsafe_allow_html=True)

# --- 1. DATA LAYER: Dummy CSV Generation ---
def get_historical_data():
    # Creating dummy data representing the last 3 years of a business
    data = {
        'Year': [2022, 2023, 2024],
        'Revenue_M': [130, 142, 150],
        'EBITDA_Margin': [0.35, 0.38, 0.40],
        'CapEx_M': [25, 30, 35]
    }
    return pd.DataFrame(data)

hist_df = get_historical_data()
base_rev = hist_df['Revenue_M'].iloc[-1]
base_margin = hist_df['EBITDA_Margin'].iloc[-1]
base_capex = hist_df['CapEx_M'].iloc[-1]

# --- 2. SIDEBAR: Strategy & Parameters ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1541/1541415.png", width=80)
    st.title("Control Center")
    
    st.subheader("🎯 Strategic Option")
    strategy = st.selectbox(
        "Select Growth Lever:",
        ["Organic Growth", "Premium Pricing", "Aggressive Expansion", "Cost Optimization"]
    )
    
    # Strategy-specific logic adjustments
    strat_mult = {"Organic Growth": 1.0, "Premium Pricing": 1.2, "Aggressive Expansion": 1.5, "Cost Optimization": 0.9}
    margin_mult = {"Organic Growth": 1.0, "Premium Pricing": 1.1, "Aggressive Expansion": 0.8, "Cost Optimization": 1.2}

    st.divider()
    
    with st.expander("Revenue & Growth", expanded=True):
        rev_val = st.slider("Target Revenue ($M)", 50, 500, int(base_rev * strat_mult[strategy]))
        growth_mu = st.slider("Expected Growth (%)", 0, 50, 10) / 100
        rev_sigma = st.slider("Revenue Volatility (%)", 5, 40, 20) / 100
        
    with st.expander("Operations & Tax", expanded=True):
        ebitda_mu = st.slider("Target Margin (%)", 10, 60, int(base_margin * margin_mult[strategy] * 100)) / 100
        tax_rate = st.slider("Effective Tax Rate (%)", 15, 35, 28) / 100
        margin_sigma = st.slider("Margin Volatility (%)", 5, 30, 15) / 100

    with st.expander("Capital & Risk", expanded=True):
        capex_val = st.slider("Investment/CapEx ($M)", 5, 100, base_capex)
        wacc = st.slider("WACC / Discount Rate (%)", 5, 20, 11) / 100
        iterations = 10000
        years = 5

# --- 3. SIMULATION ENGINE ---
@st.cache_data
def run_monte_carlo(rev, growth, r_vol, margin, m_vol, tax, cpcl, disc, iters, yrs):
    np.random.seed(42)
    # Simulate paths
    growth_paths = np.random.normal(growth, r_vol, (iters, yrs))
    margin_paths = np.random.normal(margin, m_vol, (iters, yrs))
    
    results = []
    for i in range(iters):
        revenues = [rev]
        for y in range(yrs - 1):
            revenues.append(revenues[-1] * (1 + growth_paths[i, y]))
        
        ebitda = np.array(revenues) * margin_paths[i]
        fcf = ebitda * (1 - tax)
        
        discounts = [(1 + disc)**t for t in range(1, yrs + 1)]
        npv = np.sum(fcf / discounts) - cpcl
        results.append(npv)
        
    return np.array(results), growth_paths, margin_paths

npvs, g_paths, m_paths = run_monte_carlo(rev_val, growth_mu, rev_sigma, ebitda_mu, margin_sigma, tax_rate, capex_val, wacc, iterations, years)

# --- 4. DASHBOARD UI ---
st.title("📊 Strategic Business Simulator")
st.markdown(f"**Current Strategy:** `{strategy}` | **Baseline:** Based on 2024 Historical Data")

# KPI CARDS
p90_npv = np.percentile(npvs, 10)
mean_npv = np.mean(npvs)
var_95 = np.percentile(npvs, 5)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f'<div class="metric-card"><p class="metric-label">P90 NPV (Safe Bet)</p><p class="metric-value">${p90_npv:,.1f}M</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><p class="metric-label">Expected NPV</p><p class="metric-value">${mean_npv:,.1f}M</p></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card" style="border-top: 4px solid #d9534f;"><p class="metric-label">95% Value at Risk</p><p class="metric-value">${var_95:,.1f}M</p></div>', unsafe_allow_html=True)

st.write("---")

# CHARTS GRID
col_left, col_right = st.columns(2)

with col_left:
    # 1. Histogram
    fig1 = px.histogram(npvs, nbins=60, title="Probability Distribution of NPV", 
                        color_discrete_sequence=['#0047AB'], opacity=0.8)
    fig1.add_vline(x=p90_npv, line_dash="dash", line_color="orange", annotation_text="P90")
    fig1.add_vline(x=0, line_color="black", annotation_text="Break Even")
    st.plotly_chart(fig1, use_container_width=True)

    # 2. Percentile Box Plot
    fig2 = px.box(npvs, points=False, title="NPV Range & Quartiles", orientation='h', color_discrete_sequence=['#2ecc71'])
    st.plotly_chart(fig2, use_container_width=True)

with col_right:
    # 3. Risk-Reward Scatter (Growth vs NPV)
    avg_g = g_paths.mean(axis=1)
    fig3 = px.scatter(x=avg_g, y=npvs, title="Sensitivity: Avg Growth vs NPV", 
                      labels={'x': 'Mean Annual Growth', 'y': 'Resulting NPV ($M)'}, opacity=0.3)
    st.plotly_chart(fig3, use_container_width=True)

    # 4. Correlation Heatmap
    corr_df = pd.DataFrame({'NPV': npvs, 'Growth': avg_g, 'Margin': m_paths.mean(axis=1)})
    fig4 = px.imshow(corr_df.corr(), text_auto=True, title="Variable Correlation Matrix", color_continuous_scale='Blues')
    st.plotly_chart(fig4, use_container_width=True)

# --- 5. HISTORICAL REFERENCE TABLE ---
with st.expander("📂 View Historical Baseline Data (Dummy CSV)"):
    st.table(hist_df)
    csv = hist_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Historical Data", data=csv, file_name="history.csv", mime="text/csv")
