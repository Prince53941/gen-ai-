import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="Executive Decision Suite", layout="wide")

# --- PROFESSIONAL STYLING ---
st.markdown("""
<style>
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e1e4e8;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-label { font-size: 14px; color: #586069; font-weight: bold; text-transform: uppercase; }
    .metric-value { font-size: 32px; color: #0047AB; font-weight: 800; }
    .guide-box {
        background-color: #f0f7ff;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #0047AB;
        margin-bottom: 20px;
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

# --- 2. SIDEBAR (YOUR EXISTING CONTROLS) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1541/1541415.png", width=60)
    st.title("Strategic Settings")
    strategy = st.selectbox("Strategy:", ["Organic Growth", "Premium Pricing", "Aggressive Expansion", "Cost Optimization"])
    
    with st.expander("Revenue & Growth", expanded=True):
        rev_val = st.slider("Starting Revenue ($M)", 50, 500, 150)
        growth_mu = st.slider("Expected Growth (%)", 0, 50, 10) / 100
        rev_sigma = st.slider("Revenue Volatility (%)", 5, 40, 20) / 100
    with st.expander("Costs & Risks", expanded=True):
        ebitda_mu = st.slider("Profit Margin (%)", 10, 60, 40) / 100
        tax_rate = st.slider("Tax Rate (%)", 15, 35, 28) / 100
        wacc = st.slider("WACC (Risk) (%)", 5, 20, 11) / 100
        capex_val = st.slider("Investment ($M)", 5, 100, 35)
        years = 5

# --- 3. SIMULATION ENGINE ---
@st.cache_data
def simulate(rev, growth, r_vol, margin, tax, cpcl, disc, yrs, iters=10000):
    np.random.seed(42)
    g_paths = np.random.normal(growth, r_vol, (iters, yrs))
    all_npvs = []
    for i in range(iters):
        revs = [rev]
        for y in range(yrs - 1): revs.append(revs[-1] * (1 + g_paths[i, y]))
        fcf = (np.array(revs) * margin) * (1 - tax)
        npv = np.sum([fcf[t] / (1 + disc)**(t+1) for t in range(yrs)]) - cpcl
        all_npvs.append(npv)
    return np.array(all_npvs)

npvs = simulate(rev_val, growth_mu, rev_sigma, ebitda_mu, tax_rate, capex_val, wacc, years)

# --- 4. DASHBOARD UI ---
st.title("Strategic Investment Decision Board")
st.markdown(f"**Current Path:** `{strategy}` | **Time Horizon:** {years} Years")

# TOP LEVEL KPI CARDS
mean_profit = np.mean(npvs)
success_rate = (npvs > 0).sum() / 10000 * 100
safety_floor = np.percentile(npvs, 10) # P10 (Conservative)

c1, c2, c3 = st.columns(3)
with c1: st.markdown(f'<div class="metric-card"><p class="metric-label">Expected Total Profit</p><p class="metric-value">${mean_profit:,.1f}M</p></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="metric-card"><p class="metric-label">Chance of Profit</p><p class="metric-value">{success_rate:.1f}%</p></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="metric-card"><p class="metric-label">Safety Floor (Conservative)</p><p class="metric-value">${safety_floor:,.1f}M</p></div>', unsafe_allow_html=True)

st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("""<div class="guide-box"><b>Decision 1: The Confidence Test</b><br>
    Is the <b>Green area</b> significantly larger than the Red? 
    If the Chance of Profit is below 80%, consider lowering your Initial Investment or increasing Margins.</div>""", unsafe_allow_html=True)
    
    # CHART 1: SIMPLE PROFIT vs LOSS DISTRIBUTION
    fig1 = px.histogram(npvs, nbins=50, title="Profit & Loss Scenarios", 
                        color=(npvs > 0), color_discrete_map={True: '#28a745', False: '#dc3545'},
                        labels={'value': 'Total Project Value ($M)', 'count': 'Likelihood'})
    fig1.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', 
                      xaxis_title="Final Profit/Loss after 5 Years ($M)", yaxis_title="Probability Level")
    st.plotly_chart(fig1, use_container_width=True)

with col_right:
    st.markdown("""<div class="guide-box"><b>Decision 2: The Stress Test</b><br>
    How does Profit change if <b>Sales Growth</b> or <b>Interest Rates</b> shift? 
    Look for your "current setup" in the middle. If a small drop in growth turns the box Red, the plan is too fragile.</div>""", unsafe_allow_html=True)
    
    # CHART 2: STRATEGIC HEATMAP (DECISION MATRIX)
    w_range = np.linspace(wacc*0.7, wacc*1.3, 5)
    g_range = np.linspace(growth_mu*0.7, growth_mu*1.3, 5)
    sens_matrix = [[(rev_val * (1+g) * ebitda_mu) / (w) - capex_val for w in w_range] for g in g_range]
    
    fig2 = px.imshow(sens_matrix, 
                     x=[f"{int(x*100)}% Risk" for x in w_range], 
                     y=[f"{int(y*100)}% Growth" for y in g_range],
                     text_auto='.1f', title="Strategy Performance Matrix",
                     color_continuous_scale='RdYlGn', aspect="auto")
    fig2.update_xaxes(side="bottom", title="Market Risk (Interest Rates/WACC)")
    fig2.update_yaxes(title="Annual Sales Growth")
    st.plotly_chart(fig2, use_container_width=True)

# HISTORICAL FOOTER
with st.expander("View Historical 4-Year Baseline"):
    st.line_chart(hist_df.set_index('Month')['Revenue_M'])
