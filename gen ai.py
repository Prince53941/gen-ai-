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

# --- 1. DATA LAYER ---
@st.cache_data
def get_historical_data():
    dates = pd.date_range(end=datetime.today(), periods=48, freq='M')
    np.random.seed(42)
    growth_trend = np.linspace(10.0, 14.0, 48) 
    noise = np.random.normal(1, 0.05, 48)
    rev_m = growth_trend * noise
    margin_m = np.random.normal(0.35, 0.03, 48)
    capex_m = np.random.normal(2.0, 0.4, 48)
    return pd.DataFrame({'Month': dates, 'Revenue_M': rev_m, 'EBITDA_Margin': margin_m, 'CapEx_M': capex_m})

hist_df = get_historical_data()
last_12m = hist_df.tail(12)
base_rev = last_12m['Revenue_M'].sum()
base_margin = last_12m['EBITDA_Margin'].mean()
base_capex = last_12m['CapEx_M'].sum()

# --- 2. SIDEBAR (KEPT EXACTLY AS PROVIDED) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1541/1541415.png", width=80)
    st.title("Control Center")
    strategy = st.selectbox("Select Growth Lever:", ["Organic Growth", "Premium Pricing", "Aggressive Expansion", "Cost Optimization"])
    
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

# --- 3. SIMULATION ENGINE ---
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

p90_npv = np.percentile(npvs, 10)
mean_npv = np.mean(npvs)
var_95 = np.percentile(npvs, 5)
risk_adj_return = mean_npv / np.std(npvs)

c1, c2, c3, c4 = st.columns(4)
with c1: st.markdown(f'<div class="metric-card"><p class="metric-label">P90 NPV (Low Risk)</p><p class="metric-value">${p90_npv:,.1f}M</p></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="metric-card"><p class="metric-label">Expected NPV</p><p class="metric-value">${mean_npv:,.1f}M</p></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="metric-card" style="border-top: 4px solid #d9534f;"><p class="metric-label">95% Value at Risk</p><p class="metric-value">${var_95:,.1f}M</p></div>', unsafe_allow_html=True)
with c4: st.markdown(f'<div class="metric-card" style="border-top: 4px solid #0047AB;"><p class="metric-label">Risk/Reward Ratio</p><p class="metric-value">{risk_adj_return:.2f}</p></div>', unsafe_allow_html=True)

st.write("---")

# ENHANCED DECISION TABS
tabs = st.tabs(["📉 Market Trends", "🔮 Risk Distribution", "🕹️ Sensitivity & Strategy", "🌪️ Key Drivers"])

with tabs[0]:
    col_a, col_b = st.columns([2, 1])
    with col_a:
        fig_hist = px.area(hist_df, x='Month', y='Revenue_M', title="4-Year Revenue Growth Momentum", color_discrete_sequence=['#0047AB'])
        st.plotly_chart(fig_hist, use_container_width=True)
    with col_b:
        # NEW: Waterfall Chart for Strategy Impact
        fig_water = go.Figure(go.Waterfall(
            orientation = "v",
            measure = ["relative", "relative", "total"],
            x = ["Base Revenue", "Strategic Growth", "Final Forecasted"],
            textposition = "outside",
            y = [rev_val, rev_val * growth_mu * years, rev_val * (1+growth_mu)**years],
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
        ))
        fig_water.update_layout(title="Strategic Value Bridge (5YR)")
        st.plotly_chart(fig_water, use_container_width=True)

with tabs[1]:
    cl, cr = st.columns(2)
    with cl:
        fig1 = px.histogram(npvs, nbins=60, title="NPV Probability Density", color_discrete_sequence=['#0047AB'], opacity=0.8, marginal="box")
        fig1.add_vline(x=0, line_color="red", line_width=2, annotation_text="Breakeven Loss")
        st.plotly_chart(fig1, use_container_width=True)
    with cr:
        sorted_npv = np.sort(npvs)
        p_values = np.linspace(0, 100, len(sorted_npv))
        fig2 = px.line(x=sorted_npv, y=p_values, title="Confidence in Profitability", labels={'x': 'Potential NPV ($M)', 'y': 'Probability (%)'})
        fig2.add_hrect(y0=0, y1=5, fillcolor="red", opacity=0.2, annotation_text="High Danger Zone")
        st.plotly_chart(fig2, use_container_width=True)

with tabs[2]:
    st.subheader("Boardroom Sensitivity Analysis")
    # NEW: Heatmap for WACC vs Growth
    w_range = np.linspace(wacc*0.5, wacc*1.5, 7)
    g_range = np.linspace(growth_mu*0.5, growth_mu*1.5, 7)
    # Simplified sensitivity matrix
    sens_data = [[ (rev_val * (1+g) * ebitda_mu) / (w + 0.01) - capex_val for w in w_range] for g in g_range]
    
    fig_heat = px.imshow(sens_data, 
                         x=[f"{int(x*100)}% WACC" for x in w_range],
                         y=[f"{int(y*100)}% Growth" for y in g_range],
                         text_auto='.1f', title="Decision Matrix: Growth vs. Interest Rates (NPV $M)",
                         color_continuous_scale='RdYlGn')
    st.plotly_chart(fig_heat, use_container_width=True)

with tabs[3]:
    # NEW: Tornado Chart (Impact Analysis)
    st.subheader("What Moves the Needle?")
    drivers = ['Growth Rate', 'Margin', 'WACC', 'CapEx']
    # Calculate simple impact by shifting each variable by 10%
    impacts = [
        (np.mean(run_monte_carlo(rev_val, growth_mu*1.1, rev_sigma, ebitda_mu, margin_sigma, tax_rate, capex_val, wacc, 1000, 5)[0]) - mean_npv),
        (np.mean(run_monte_carlo(rev_val, growth_mu, rev_sigma, ebitda_mu*1.1, margin_sigma, tax_rate, capex_val, wacc, 1000, 5)[0]) - mean_npv),
        (np.mean(run_monte_carlo(rev_val, growth_mu, rev_sigma, ebitda_mu, margin_sigma, tax_rate, capex_val, wacc*1.1, 1000, 5)[0]) - mean_npv),
        (np.mean(run_monte_carlo(rev_val, growth_mu, rev_sigma, ebitda_mu, margin_sigma, tax_rate, capex_val*1.1, wacc, 1000, 5)[0]) - mean_npv)
    ]
    fig_torn = px.bar(x=impacts, y=drivers, orientation='h', title="Tornado Analysis: Impact of 10% Variable Shift on Mean NPV",
                      color=impacts, color_continuous_scale='RdBu')
    st.plotly_chart(fig_torn, use_container_width=True)

with st.expander("📂 View Full 48-Month Historical Data"):
    st.dataframe(hist_df, use_container_width=True)
    st.download_button("Download Monthly History", data=hist_df.to_csv(index=False).encode('utf-8'), file_name="monthly_history.csv", mime="text/csv")
