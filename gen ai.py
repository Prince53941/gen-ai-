import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="StratSim: Executive Decision Suite", layout="wide")

# --- CUSTOM STYLING (KEPT AS PROVIDED) ---
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

# --- 1. DATA LAYER (48 Months History) ---
@st.cache_data
def get_historical_data():
    dates = pd.date_range(end=datetime.today(), periods=48, freq='M')
    np.random.seed(42)
    rev_m = np.linspace(10.0, 14.0, 48) * np.random.normal(1, 0.05, 48)
    margin_m = np.random.normal(0.35, 0.03, 48)
    return pd.DataFrame({'Month': dates, 'Revenue_M': rev_m, 'EBITDA_Margin': margin_m})

hist_df = get_historical_data()
base_rev = hist_df.tail(12)['Revenue_M'].sum()
base_margin = hist_df.tail(12)['EBITDA_Margin'].mean()

# --- 2. SIDEBAR (KEPT AS PROVIDED) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1541/1541415.png", width=80)
    st.title("Control Center")
    strategy = st.selectbox("Select Growth Lever:", ["Organic Growth", "Premium Pricing", "Aggressive Expansion", "Cost Optimization"])
    
    with st.expander("Revenue & Growth", expanded=True):
        rev_val = st.slider("Target Revenue ($M)", 50, 500, 150)
        growth_mu = st.slider("Expected Growth (%)", 0, 50, 10) / 100
        rev_sigma = st.slider("Revenue Volatility (%)", 5, 40, 20) / 100
    with st.expander("Operations & Tax", expanded=True):
        ebitda_mu = st.slider("Target Margin (%)", 10, 60, 40) / 100
        tax_rate = st.slider("Tax Rate (%)", 15, 35, 28) / 100
        margin_sigma = st.slider("Margin Volatility (%)", 5, 30, 15) / 100
    with st.expander("Capital & Risk", expanded=True):
        capex_val = st.slider("Investment/CapEx ($M)", 5, 100, 35)
        wacc = st.slider("WACC (%)", 5, 20, 11) / 100
        iterations = 10000
        years = 5

# --- 3. SIMULATION ENGINE ---
@st.cache_data
def run_simulation(rev, growth, r_vol, margin, m_vol, tax, cpcl, disc, iters, yrs):
    np.random.seed(42)
    growth_paths = np.random.normal(growth, r_vol, (iters, yrs))
    margin_paths = np.random.normal(margin, m_vol, (iters, yrs))
    results = []
    for i in range(iters):
        revs = [rev]
        for y in range(yrs - 1): revs.append(revs[-1] * (1 + growth_paths[i, y]))
        fcf = (np.array(revs) * margin_paths[i]) * (1 - tax)
        npv = np.sum([fcf[t] / (1 + disc)**(t+1) for t in range(yrs)]) - cpcl
        results.append(npv)
    return np.array(results)

npvs = run_simulation(rev_val, growth_mu, rev_sigma, ebitda_mu, margin_sigma, tax_rate, capex_val, wacc, iterations, years)

# --- 4. DASHBOARD UI ---
st.title("🏛️ Strategic Investment Decision Suite")

# KPI ROW
p90 = np.percentile(npvs, 10)
mean_v = np.mean(npvs)
prob_loss = (npvs < 0).sum() / iterations * 100

c1, c2, c3, c4 = st.columns(4)
with c1: st.markdown(f'<div class="metric-card"><p class="metric-label">Expected NPV</p><p class="metric-value">${mean_v:,.1f}M</p></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="metric-card"><p class="metric-label">P90 (Safety Floor)</p><p class="metric-value">${p90:,.1f}M</p></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="metric-card" style="border-top: 5px solid #d9534f;"><p class="metric-label">Prob. of Loss</p><p class="metric-value">{prob_loss:.1f}%</p></div>', unsafe_allow_html=True)
with c4: st.markdown(f'<div class="metric-card" style="border-top: 5px solid #2ecc71;"><p class="metric-label">ROI Ratio</p><p class="metric-value">{(mean_v/capex_val):.2f}x</p></div>', unsafe_allow_html=True)

st.divider()

# TABBED VISUALIZATIONS
tab1, tab2, tab3 = st.tabs(["📊 Risk Profile", "🌡️ Sensitivity Heatmap", "🌪️ Key Drivers"])

with tab1:
    col_l, col_r = st.columns(2)
    with col_l:
        # 1. ENHANCED DISTRIBUTION
        fig_dist = px.histogram(npvs, nbins=50, title="Where will the project land?", color_discrete_sequence=['#0047AB'], opacity=0.7)
        fig_dist.add_vline(x=0, line_color="red", line_width=3, annotation_text="Danger Zone")
        st.plotly_chart(fig_dist, use_container_width=True)
    with col_r:
        # 2. PROBABILITY GAUGE
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = 100 - prob_loss,
            title = {'text': "Confidence in Success (%)"},
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#0047AB"},
                     'steps': [{'range': [0, 50], 'color': "#ffcfcf"}, {'range': [50, 80], 'color': "#fff4cf"}, {'range': [80, 100], 'color': "#cfdfcf"}]}))
        st.plotly_chart(fig_gauge, use_container_width=True)

with tab2:
    # 3. STRATEGIC HEATMAP (Decision Matrix)
    st.subheader("What happens if external conditions change?")
    w_range = np.linspace(wacc*0.7, wacc*1.3, 7)
    g_range = np.linspace(growth_mu*0.7, growth_mu*1.3, 7)
    sens_matrix = [[(rev_val * (1+g) * ebitda_mu * (1-tax_rate)) / (w) - capex_val for w in w_range] for g in g_range]
    
    fig_heat = px.imshow(sens_matrix, x=[f"{int(x*100)}% WACC" for x in w_range], y=[f"{int(y*100)}% Growth" for y in g_range],
                         text_auto='.1f', title="NPV Sensitivity: Growth vs. Interest Rates", color_continuous_scale='RdYlGn')
    st.plotly_chart(fig_heat, use_container_width=True)

with tab3:
    # 4. TORNADO CHART (Impact Analysis)
    st.subheader("Which slider 'moves the needle' most?")
    drivers = ['Growth Rate', 'Margin', 'WACC', 'CapEx']
    # Calculate 10% swing impact
    impacts = [
        np.mean(run_simulation(rev_val, growth_mu*1.1, rev_sigma, ebitda_mu, margin_sigma, tax_rate, capex_val, wacc, 1000, 5)) - mean_v,
        np.mean(run_simulation(rev_val, growth_mu, rev_sigma, ebitda_mu*1.1, margin_sigma, tax_rate, capex_val, wacc, 1000, 5)) - mean_v,
        np.mean(run_simulation(rev_val, growth_mu, rev_sigma, ebitda_mu, margin_sigma, tax_rate, capex_val, wacc*1.1, 1000, 5)) - mean_v,
        (mean_v - (mean_v - (capex_val * 0.1))) - mean_v # CapEx impact
    ]
    fig_torn = px.bar(x=impacts, y=drivers, orientation='h', title="Tornado Analysis: Impact of 10% Change", color=impacts, color_continuous_scale='RdBu')
    st.plotly_chart(fig_torn, use_container_width=True)
