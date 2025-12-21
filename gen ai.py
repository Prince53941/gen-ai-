import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="StratSim: Executive Decision Suite", layout="wide")

# --- CUSTOM STYLING (KEPT EXACTLY AS PROVIDED) ---
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

# --- 1. DATA LAYER: Extensive Monthly Historical Data (48 Months) ---
@st.cache_data
def get_historical_data():
    # Creating 4 years (48 months) of historical data for deeper analysis
    dates = pd.date_range(end=datetime.today(), periods=48, freq='M')
    np.random.seed(42)
    
    # Simulate a business with 40% growth over 4 years and slight seasonality
    growth_trend = np.linspace(10.0, 14.0, 48) 
    noise = np.random.normal(1, 0.05, 48)
    
    rev_m = growth_trend * noise
    margin_m = np.random.normal(0.35, 0.03, 48)
    capex_m = np.random.normal(2.0, 0.4, 48)
    
    data = {
        'Month': dates,
        'Revenue_M': rev_m,
        'EBITDA_Margin': margin_m,
        'CapEx_M': capex_m
    }
    return pd.DataFrame(data)

hist_df = get_historical_data()
# Calculations based on the last 12 months for the sliders
last_12m = hist_df.tail(12)
base_rev = last_12m['Revenue_M'].sum()
base_margin = last_12m['EBITDA_Margin'].mean()
base_capex = last_12m['CapEx_M'].sum()

# --- 2. SIDEBAR: Strategy & Parameters (KEPT EXACTLY AS PROVIDED) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1541/1541415.png", width=80)
    st.title("Control Center")
    
    st.subheader("🎯 Strategic Option")
    strategy = st.selectbox(
        "Select Growth Lever:",
        ["Organic Growth", "Premium Pricing", "Aggressive Expansion", "Cost Optimization"]
    )
    
    strat_mult = {"Organic Growth": 1.0, "Premium Pricing": 1.2, "Aggressive Expansion": 1.5, "Cost Optimization": 0.9}
    margin_mult = {"Organic Growth": 1.0, "Premium Pricing": 1.1, "Aggressive Expansion": 0.8, "Cost Optimization": 1.2}

    st.divider()
    
    with st.expander("Revenue & Growth", expanded=True):
        rev_val = st.slider("Target Revenue ($M)", 50, 500, int(base_rev * strat_mult[strategy]))
        growth_mu = st.slider("Expected Growth (%)", 0, 50, 12) / 100
        rev_sigma = st.slider("Revenue Volatility (%)", 5, 40, 20) / 100
        
    with st.expander("Operations & Tax", expanded=True):
        ebitda_mu = st.slider("Target Margin (%)", 10, 60, int(base_margin * margin_mult[strategy] * 100)) / 100
        tax_rate = st.slider("Effective Tax Rate (%)", 15, 35, 25) / 100
        margin_sigma = st.slider("Margin Volatility (%)", 5, 30, 12) / 100

    with st.expander("Capital & Risk", expanded=True):
        capex_val = st.slider("Investment/CapEx ($M)", 5, 100, int(base_capex))
        wacc = st.slider("WACC / Discount Rate (%)", 5, 20, 10) / 100
        iterations = 10000
        years = 5

# --- 3. SIMULATION ENGINE (KEPT AS PROVIDED) ---
@st.cache_data
def run_monte_carlo(rev, growth, r_vol, margin, m_vol, tax, cpcl, disc, iters, yrs):
    np.random.seed(42)
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
st.markdown(f"**Current Strategy:** `{strategy}` | **Baseline:** Based on 4-Year Monthly Historical Data")

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

# ENHANCED CHARTS GRID
tabs = st.tabs(["📉 Historical Analysis", "🔮 Forecast Distribution", "🎯 Risk Metrics"])

with tabs[0]:
    st.subheader("4-Year Performance Trends")
    # Area Chart for Revenue is much clearer for growth visualization
    fig_hist = px.area(hist_df, x='Month', y='Revenue_M', title="Monthly Revenue Momentum ($M)",
                       color_discrete_sequence=['#0047AB'])
    fig_hist.update_layout(hovermode="x unified")
    st.plotly_chart(fig_hist, use_container_width=True)

with tabs[1]:
    col_left, col_right = st.columns(2)
    with col_left:
        # 1. Distribution Chart with Probability Density
        fig1 = px.histogram(npvs, nbins=60, title="Probability Distribution of NPV", 
                            color_discrete_sequence=['#0047AB'], opacity=0.8, marginal="violin")
        fig1.add_vline(x=p90_npv, line_dash="dash", line_color="orange", annotation_text="P90")
        fig1.add_vline(x=0, line_color="black", line_width=2, annotation_text="Breakeven")
        st.plotly_chart(fig1, use_container_width=True)

    with col_right:
        # 2. Cumulative Confidence Curve (Critical for Decisions)
        sorted_npv = np.sort(npvs)
        p_values = np.linspace(0, 100, len(sorted_npv))
        fig2 = px.line(x=sorted_npv, y=p_values, title="Confidence Level vs. Profitability",
                       labels={'x': 'Potential NPV ($M)', 'y': 'Confidence (%)'})
        fig2.add_hrect(y0=0, y1=10, fillcolor="red", opacity=0.1, annotation_text="Risk Zone")
        st.plotly_chart(fig2, use_container_width=True)

with tabs[2]:
    col_left, col_right = st.columns(2)
    with col_left:
        # 3. Risk-Reward Scatter (Cleaner Visualization)
        avg_g = g_paths.mean(axis=1)
        fig3 = px.scatter(x=avg_g, y=npvs, title="NPV Sensitivity to Growth Rate", 
                          labels={'x': 'Annual Growth (%)', 'y': 'NPV ($M)'}, opacity=0.4,
                          color=npvs, color_continuous_scale='RdYlGn')
        st.plotly_chart(fig3, use_container_width=True)

    with col_right:
        # 4. Correlation Heatmap (Professional Blues scale)
        corr_df = pd.DataFrame({'NPV Outcome': npvs, 'Growth Risk': avg_g, 'Margin Risk': m_paths.mean(axis=1)})
        fig4 = px.imshow(corr_df.corr(), text_auto=True, title="Key Driver Analysis (Correlation)", 
                         color_continuous_scale='Blues')
        st.plotly_chart(fig4, use_container_width=True)

# --- 5. HISTORICAL REFERENCE TABLE ---
with st.expander("📂 View Full 48-Month Historical Data"):
    st.dataframe(hist_df, use_container_width=True)
    csv = hist_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Monthly History", data=csv, file_name="monthly_history.csv", mime="text/csv")
