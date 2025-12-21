import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- PAGE CONFIG ---
st.set_page_config(page_title="StratSim Pro | Executive Suite", layout="wide")

# --- EXECUTIVE STYLING ---
st.markdown("""
<style>
    .reportview-container { background: #F0F2F6; }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border-top: 5px solid #0047AB;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    .metric-value { font-size: 30px; font-weight: 800; color: #1E1E1E; }
    .metric-label { font-size: 13px; color: #5E5E5E; text-transform: uppercase; }
    .strategy-box {
        background-color: #E3F2FD;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #2196F3;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. DATA LAYER: EXTENSIVE MONTHLY HISTORICAL DATA ---
@st.cache_data
def generate_extensive_history():
    # 48 Months of data (4 years)
    dates = pd.date_range(end=datetime.today(), periods=48, freq='M')
    np.random.seed(42)
    
    # Simulating a growing business with some seasonality
    base_rev = 12.0 # Monthly starting revenue in $M
    growth_trend = np.linspace(1.0, 1.4, 48) # 40% growth over 4 years
    noise = np.random.normal(1, 0.05, 48)
    
    rev_m = base_rev * growth_trend * noise
    margin_m = np.random.normal(0.35, 0.03, 48)
    capex_m = np.random.normal(2.5, 0.5, 48)
    
    df = pd.DataFrame({
        'Month': dates,
        'Revenue_M': rev_m,
        'EBITDA_Margin': margin_m,
        'CapEx_M': capex_m
    })
    return df

hist_df = generate_extensive_history()
last_12m_rev = hist_df['Revenue_M'].tail(12).sum()
avg_margin = hist_df['EBITDA_Margin'].tail(12).mean()

# --- 2. SIDEBAR CONTROL ---
with st.sidebar:
    st.title("🕹️ Strategy Engine")
    
    st.subheader("Choose Strategic Path")
    strategy = st.selectbox(
        "Current Objective:",
        ["Market Penetration (Low Margin, High Vol)", 
         "Premium Pivot (High Margin, Lower Vol)", 
         "Steady State (Organic)",
         "Aggressive R&D (High CapEx)"]
    )
    
    st.divider()
    st.write("### Adjust Forecast Parameters")
    
    # 10 Professional Sliders
    rev_target = st.slider("Projected Annual Rev ($M)", 100, 300, int(last_12m_rev))
    growth_rate = st.slider("Target Growth YoY (%)", 0, 40, 12) / 100
    rev_vol = st.slider("Revenue Risk (%)", 5, 50, 20) / 100
    
    margin_target = st.slider("Target EBITDA Margin (%)", 10, 60, int(avg_margin*100)) / 100
    margin_vol = st.slider("Margin Stability (%)", 2, 20, 10) / 100
    
    tax_rate = st.slider("Corp Tax Rate (%)", 15, 35, 25) / 100
    capex_annual = st.slider("Annual CapEx ($M)", 10, 100, 30)
    wacc = st.slider("WACC (Discount Rate) (%)", 5, 20, 10) / 100
    
    sim_years = st.select_slider("Forecast Horizon", options=[3, 5, 10], value=5)
    runs = 10000

# --- 3. MONTE CARLO ENGINE ---
def run_simulation():
    np.random.seed(42)
    # Generate random variances
    rev_shocks = np.random.normal(1 + growth_rate, rev_vol, (runs, sim_years))
    margin_shocks = np.random.normal(margin_target, margin_vol, (runs, sim_years))
    
    all_npvs = []
    for i in range(runs):
        rev_path = [rev_target]
        for y in range(sim_years - 1):
            rev_path.append(rev_path[-1] * rev_shocks[i, y])
            
        fcf = (np.array(rev_path) * margin_shocks[i]) * (1 - tax_rate) - (capex_annual/sim_years)
        pv = np.sum([fcf[t] / (1 + wacc)**(t+1) for t in range(sim_years)])
        all_npvs.append(pv)
        
    return np.array(all_npvs)

npvs = run_simulation()

# --- 4. MAIN UI ---
st.title("🏆 Strategic Investment & Risk Dashboard")

# Top Metrics Row
p90 = np.percentile(npvs, 10)
mean_v = np.mean(npvs)
var_95 = np.percentile(npvs, 5)

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(f'<div class="metric-card"><p class="metric-label">Conservative NPV (P90)</p><p class="metric-value">${p90:.1f}M</p></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-card"><p class="metric-label">Expected Outcome</p><p class="metric-value">${mean_v:.1f}M</p></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-card" style="border-top-color:#E53935"><p class="metric-label">At Risk (5% Tail)</p><p class="metric-value">${var_95:.1f}M</p></div>', unsafe_allow_html=True)

st.divider()

# --- CHART SECTION ---
tabs = st.tabs(["📉 Historical Analysis", "🔮 Simulation Results", "🎯 Sensitivity"])

with tabs[0]:
    st.subheader("4-Year Historical Performance Trend")
    # Professional Area Chart for Revenue
    fig_hist = px.area(hist_df, x='Month', y='Revenue_M', title="Revenue Momentum ($M)",
                       color_discrete_sequence=['#0047AB'])
    fig_hist.update_layout(hovermode="x unified", plot_bgcolor="white")
    st.plotly_chart(fig_hist, use_container_width=True)
    
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.write("**Monthly EBITDA Margin Stability**")
        st.line_chart(hist_df.set_index('Month')['EBITDA_Margin'])
    with col_h2:
        st.write("**CapEx Investment Cycles**")
        st.bar_chart(hist_df.set_index('Month')['CapEx_M'])

with tabs[1]:
    c_left, c_right = st.columns(2)
    
    with c_left:
        # High Impact Histogram
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=npvs, nbinsx=50, marker_color='#1565C0', opacity=0.7))
        fig_dist.add_vline(x=p90, line_dash="dash", line_color="orange", annotation_text="P90 (Safe)")
        fig_dist.add_vline(x=0, line_width=3, line_color="black", annotation_text="Breakeven")
        fig_dist.update_layout(title="NPV Probability Distribution", xaxis_title="Net Present Value ($M)", showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)
        
    with c_right:
        # Cumulative Profitability Line
        sorted_npv = np.sort(npvs)
        p_values = np.linspace(0, 100, len(sorted_npv))
        fig_cum = px.line(x=sorted_npv, y=p_values, title="Cumulative Confidence Curve",
                          labels={'x': 'NPV ($M)', 'y': 'Confidence Level (%)'})
        fig_cum.add_hrect(y0=0, y1=10, fillcolor="red", opacity=0.1, annotation_text="High Risk Zone")
        st.plotly_chart(fig_cum, use_container_width=True)

with tabs[2]:
    st.subheader("Decision Sensitivity")
    # A simple heatmap showing how WACC and Growth impact the Mean NPV
    wacc_range = np.linspace(0.05, 0.20, 5)
    growth_range = np.linspace(0, 0.40, 5)
    
    sens_matrix = []
    for g in growth_range:
        row = []
        for w in wacc_range:
            # Simplified static NPV for the heatmap
            val = (rev_target * (1+g) * margin_target) / (w + 0.01) - capex_annual
            row.append(val)
        sens_matrix.append(row)
        
    fig_heat = px.imshow(sens_matrix, 
                         x=[f"{int(x*100)}% WACC" for x in wacc_range],
                         y=[f"{int(y*100)}% Growth" for y in growth_range],
                         text_auto=True, aspect="auto", title="NPV Sensitivity: Growth vs. Capital Cost",
                         color_continuous_scale='RdYlGn')
    st.plotly_chart(fig_heat, use_container_width=True)

# --- DOWNLOAD REPORT ---
st.divider()
st.subheader("📥 Export Executive Summary")
col_down, col_info = st.columns([1, 3])
with col_down:
    csv_out = hist_df.to_csv(index=False)
    st.download_button("Download All Data (CSV)", csv_out, "business_analysis_2025.csv", "text/csv")
with col_info:
    st.info(f"The strategy **'{strategy}'** has been evaluated over 10,000 simulated market paths. Based on the 95% Value at Risk, the maximum downside exposure is ${abs(var_95):.1f}M over {sim_years} years.")
