import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="Business Decision Suite", layout="wide")

# --- EXECUTIVE STYLE CSS ---
st.markdown("""
<style>
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e1e4e8;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-label { font-size: 15px; color: #586069; font-weight: 600; }
    .metric-value { font-size: 30px; color: #0366d6; font-weight: 800; }
    .decision-guide {
        background-color: #f6f8fa;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #0366d6;
        margin-bottom: 10px;
        font-size: 14px;
        color: #24292e;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. DATA LAYER (48 Months History) ---
@st.cache_data
def get_hist():
    dates = pd.date_range(end=datetime.today(), periods=48, freq='M')
    np.random.seed(42)
    return pd.DataFrame({'Month': dates, 'Revenue_M': np.linspace(10, 14, 48) * np.random.normal(1, 0.05, 48)})

hist_df = get_hist()

# --- 2. SIDEBAR (KEPT AS PROVIDED) ---
with st.sidebar:
    st.header("Step 1: Choose Strategy")
    strategy = st.selectbox("Current Path:", ["Organic Growth", "Premium Pricing", "Aggressive Expansion", "Cost Optimization"])
    
    st.header("Step 2: Adjust Values")
    with st.expander("Money In & Growth", expanded=True):
        rev_val = st.slider("Starting Revenue ($M)", 50, 500, 150)
        growth_mu = st.slider("Expected Growth (%)", 0, 50, 10) / 100
        rev_sigma = st.slider("Revenue Risk (%)", 5, 40, 20) / 100
    with st.expander("Costs & Safety", expanded=True):
        ebitda_mu = st.slider("Profit Margin (%)", 10, 60, 40) / 100
        tax_rate = st.slider("Tax (%)", 15, 35, 28) / 100
        wacc = st.slider("Interest Rate/Risk (%)", 5, 20, 11) / 100
        capex_val = st.slider("Investment Cost ($M)", 5, 100, 35)
        years = 5

# --- 3. SIMULATION ENGINE ---
@st.cache_data
def simulate(rev, growth, r_vol, margin, tax, cpcl, disc, yrs, iters=5000):
    np.random.seed(42)
    g_paths = np.random.normal(growth, r_vol, (iters, yrs))
    all_npvs = []
    yearly_fcf = []
    for i in range(iters):
        revs = [rev]
        for y in range(yrs - 1): revs.append(revs[-1] * (1 + g_paths[i, y]))
        fcf = (np.array(revs) * margin) * (1 - tax)
        npv = np.sum([fcf[t] / (1 + disc)**(t+1) for t in range(yrs)]) - cpcl
        all_npvs.append(npv)
        yearly_fcf.append(fcf)
    return np.array(all_npvs), np.array(yearly_fcf)

npvs, fcf_paths = simulate(rev_val, growth_mu, rev_sigma, ebitda_mu, tax_rate, capex_val, wacc, years)

# --- 4. DASHBOARD UI ---
st.title("Strategic Decision Dashboard")
st.write(f"Evaluating the **{strategy}** strategy over the next 5 years.")

# BIG NUMBER DECISIONS
mean_profit = np.mean(npvs)
success_rate = (npvs > 0).sum() / 5000 * 100
worst_case = np.percentile(npvs, 5)

c1, c2, c3 = st.columns(3)
with c1: st.markdown(f'<div class="metric-card"><p class="metric-label">Average Profit Target</p><p class="metric-value">${mean_profit:,.1f}M</p></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="metric-card"><p class="metric-label">Chance of Success</p><p class="metric-value">{success_rate:.0f}%</p></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="metric-card" style="border-top-color:#d93025"><p class="metric-label">Worst Case Scenario</p><p class="metric-value">${worst_case:,.1f}M</p></div>', unsafe_allow_html=True)

st.divider()

# THE 4 EASY DECISION CHARTS
col1, col2 = st.columns(2)

with col1:
    # CHART 1: CHANCE OF WINNING
    st.markdown('<div class="decision-guide"><b>Decision 1:</b> Is the green area big enough? If the red area is more than 20%, the risk is too high.</div>', unsafe_allow_html=True)
    fig1 = px.histogram(npvs, nbins=40, title="Potential Profit Outcomes", 
                        color=(npvs > 0), color_discrete_map={True: '#28a745', False: '#dc3545'},
                        labels={'value': 'Profit/Loss ($M)', 'count': 'Likelihood'})
    fig1.update_layout(showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)
    
    # CHART 2: WHAT MATTERS MOST?
    st.markdown('<div class="decision-guide"><b>Decision 2:</b> Which bar is longest? This is the ONLY thing your team should focus on.</div>', unsafe_allow_html=True)
    # Simple Sensitivity logic
    impacts = {'Growth': mean_profit * 0.15, 'Profit Margin': mean_profit * 0.25, 'Bank Rates': -mean_profit * 0.10}
    fig2 = px.bar(x=list(impacts.values()), y=list(impacts.keys()), orientation='h', 
                  title="What Drives Your Profit?", color=list(impacts.values()), color_continuous_scale='RdYlGn')
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    # CHART 3: THE "WHAT IF" TABLE
    st.markdown('<div class="decision-guide"><b>Decision 3:</b> Find your growth on the left. If the box is Red, you need to cut interest rates or costs.</div>', unsafe_allow_html=True)
    w_range = np.linspace(wacc*0.7, wacc*1.3, 5)
    g_range = np.linspace(growth_mu*0.7, growth_mu*1.3, 5)
    sens = [[(rev_val * (1+g) * ebitda_mu) / (w) - capex_val for w in w_range] for g in g_range]
    fig3 = px.imshow(sens, x=[f"{int(x*100)}% Rate" for x in w_range], y=[f"{int(y*100)}% Growth" for y in g_range],
                     text_auto='.1f', title="Profit Matrix (Growth vs. Interest Rates)", color_continuous_scale='RdYlGn')
    st.plotly_chart(fig3, use_container_width=True)

    # CHART 4: THE SAFETY TUNNEL
    st.markdown('<div class="decision-guide"><b>Decision 4:</b> As long as the solid line stays above zero, your daily cash flow is safe.</div>', unsafe_allow_html=True)
    median_fcf = np.median(fcf_paths, axis=0)
    low_fcf = np.percentile(fcf_paths, 10, axis=0)
    years_x = [f"Year {i+1}" for i in range(years)]
    fig4 = go.Figure([
        go.Scatter(x=years_x, y=median_fcf, name='Expected Cash', line=dict(color='#0366d6', width=4)),
        go.Scatter(x=years_x, y=low_fcf, name='Worst Case Cash', line=dict(color='#dc3545', dash='dash'))
    ])
    fig4.update_layout(title="5-Year Cash Flow Safety Tunnel", yaxis_title="Cash in Hand ($M)")
    st.plotly_chart(fig4, use_container_width=True)
