import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="Executive Decision Board", layout="wide")

# --- CLEAN EXECUTIVE STYLING ---
st.markdown("""
<style>
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e1e4e8;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-label { font-size: 14px; color: #586069; font-weight: 600; text-transform: uppercase; }
    .metric-value { font-size: 32px; color: #0366d6; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

# --- 1. DATA LAYER (4-Year Monthly History) ---
@st.cache_data
def get_historical_data():
    dates = pd.date_range(end=datetime.today(), periods=48, freq='M')
    np.random.seed(42)
    # Simulating a business with 40% growth over 4 years
    rev_m = np.linspace(10.0, 14.0, 48) * np.random.normal(1, 0.05, 48)
    margin_m = np.random.normal(0.35, 0.02, 48)
    return pd.DataFrame({'Month': dates, 'Revenue_M': rev_m, 'EBITDA_Margin': margin_m})

hist_df = get_historical_data()
base_rev = hist_df.tail(12)['Revenue_M'].sum()
base_margin = hist_df.tail(12)['EBITDA_Margin'].mean()

# --- 2. SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("Settings")
    strategy = st.selectbox("Strategy:", ["Organic Growth", "Premium Pricing", "Expansion", "Cost Cutting"])
    
    st.subheader("Revenue & Growth")
    rev_val = st.slider("Target Revenue ($M)", 50, 500, int(base_rev))
    growth_mu = st.slider("Expected Growth (%)", 0, 50, 10) / 100
    rev_sigma = st.slider("Market Risk/Volatility (%)", 5, 40, 20) / 100
    
    st.subheader("Profit & Costs")
    ebitda_mu = st.slider("Profit Margin (%)", 10, 60, int(base_margin*100)) / 100
    wacc = st.slider("Interest Rate (WACC) (%)", 5, 20, 11) / 100
    capex_val = st.slider("Total Investment ($M)", 5, 100, 35)
    
    tax_rate = 0.25
    years = 5

# --- 3. SIMULATION ENGINE ---
@st.cache_data
def run_simulation(rev, growth, r_vol, margin, cpcl, disc, yrs):
    np.random.seed(42)
    iters = 10000
    g_paths = np.random.normal(growth, r_vol, (iters, yrs))
    all_npvs = []
    for i in range(iters):
        revs = [rev]
        for y in range(yrs - 1):
            revs.append(revs[-1] * (1 + g_paths[i, y]))
        fcf = (np.array(revs) * margin) * (1 - tax_rate)
        npv = np.sum([fcf[t] / (1 + disc)**(t+1) for t in range(yrs)]) - cpcl
        all_npvs.append(npv)
    return np.array(all_npvs)

npvs = run_simulation(rev_val, growth_mu, rev_sigma, ebitda_mu, capex_val, wacc, years)

# --- 4. MAIN DASHBOARD UI ---
st.title("Strategic Decision Board")
st.markdown(f"**Strategy Focus:** {strategy} | **Analysis Period:** 5 Years")

# TOP METRICS
mean_profit = np.mean(npvs)
success_rate = (npvs > 0).sum() / 10000 * 100
worst_case = np.percentile(npvs, 10)

m1, m2, m3 = st.columns(3)
with m1: st.markdown(f'<div class="metric-card"><p class="metric-label">Avg. Expected Profit</p><p class="metric-value">${mean_profit:,.1f}M</p></div>', unsafe_allow_html=True)
with m2: st.markdown(f'<div class="metric-card"><p class="metric-label">Chance of Success</p><p class="metric-value">{success_rate:.1f}%</p></div>', unsafe_allow_html=True)
with m3: st.markdown(f'<div class="metric-card"><p class="metric-label">Safety Floor (Worst Case)</p><p class="metric-value">${worst_case:,.1f}M</p></div>', unsafe_allow_html=True)

st.divider()

col_left, col_right = st.columns(2)

with col_left:
    # CHART 1: PROFIT VS LOSS
    # Decision Logic: Green = Go, Red = Risk.
    fig1 = px.histogram(
        npvs, nbins=50, 
        title="<b>DECISION 1: THE CONFIDENCE TEST</b><br><sup>Is the Green area much larger than Red? If Success is < 80%, the risk is high.</sup>",
        color=(npvs > 0), 
        color_discrete_map={True: '#28a745', False: '#dc3545'},
        labels={'value': 'Total Profit/Loss ($M)', 'count': 'Number of Scenarios'}
    )
    fig1.add_vline(x=0, line_width=2, line_color="black")
    fig1.update_layout(showlegend=False, title_font_size=18)
    st.plotly_chart(fig1, use_container_width=True)

with col_right:
    # CHART 2: STRATEGY MATRIX
    # Decision Logic: Find the center. If moving down/left turns it Red, the plan is fragile.
    w_range = np.linspace(wacc*0.7, wacc*1.3, 5)
    g_range = np.linspace(growth_mu*0.7, growth_mu*1.3, 5)
    sens = [[(rev_val * (1+g) * ebitda_mu * (1-tax_rate)) / (w) - capex_val for w in w_range] for g in g_range]
    
    fig2 = px.imshow(
        sens, 
        x=[f"{int(x*100)}% Risk" for x in w_range], 
        y=[f"{int(y*100)}% Growth" for y in g_range],
        text_auto='.1f', 
        title="<b>DECISION 2: THE STRESS TEST</b><br><sup>If a drop in Growth (moving down) turns boxes Red, the plan is too fragile.</sup>",
        color_continuous_scale='RdYlGn', 
        aspect="auto"
    )
    fig2.update_layout(title_font_size=18)
    st.plotly_chart(fig2, use_container_width=True)

# HISTORICAL REFERENCE
with st.expander("Show 4-Year Historical Baseline"):
    st.line_chart(hist_df.set_index('Month')['Revenue_M'])
