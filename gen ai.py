import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="StratSim: Decision Intelligence", layout="wide")

# --- CUSTOM STYLING (KEPT AS PROVIDED) ---
st.markdown("""
<style>
    .main { background-color: #fcfcfc; }
    .metric-card {
        background-color: white;
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #ececec;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        text-align: center;
    }
    .metric-label { font-size: 14px; color: #6e6e6e; text-transform: uppercase; font-weight: bold; }
    .metric-value { font-size: 32px; color: #002D62; font-weight: 800; margin-top: 10px; }
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
    st.image("https://cdn-icons-png.flaticon.com/512/1541/1541415.png", width=60)
    st.title("Strategic Controls")
    strategy = st.selectbox("Strategic Lever:", ["Organic Growth", "Premium Pricing", "Aggressive Expansion", "Cost Optimization"])
    
    with st.expander("Revenue & Growth", expanded=True):
        rev_val = st.slider("Initial Revenue ($M)", 50, 500, 150)
        growth_mu = st.slider("Expected Growth (%)", 0, 50, 10) / 100
        rev_sigma = st.slider("Rev Volatility (%)", 5, 40, 20) / 100
    with st.expander("Operations & Tax", expanded=True):
        ebitda_mu = st.slider("Target Margin (%)", 10, 60, 40) / 100
        tax_rate = st.slider("Tax Rate (%)", 15, 35, 28) / 100
        margin_sigma = st.slider("Margin Volatility (%)", 5, 30, 15) / 100
    with st.expander("Capital & Risk", expanded=True):
        capex_val = st.slider("CapEx ($M)", 5, 100, 35)
        wacc = st.slider("WACC (%)", 5, 20, 11) / 100
        years = 5

# --- 3. SIMULATION ENGINE ---
@st.cache_data
def simulate(rev, growth, r_vol, margin, m_vol, tax, cpcl, disc, yrs, iters=10000):
    np.random.seed(42)
    g_paths = np.random.normal(growth, r_vol, (iters, yrs))
    m_paths = np.random.normal(margin, m_vol, (iters, yrs))
    
    all_npvs = []
    yearly_data = [] # For the Fan Chart
    
    for i in range(iters):
        revs = [rev]
        for y in range(yrs - 1):
            revs.append(revs[-1] * (1 + g_paths[i, y]))
        
        fcf = (np.array(revs) * m_paths[i]) * (1 - tax)
        npv = np.sum([fcf[t] / (1 + disc)**(t+1) for t in range(yrs)]) - cpcl
        all_npvs.append(npv)
        yearly_data.append(fcf)
        
    return np.array(all_npvs), np.array(yearly_data)

npvs, yearly_fcf = simulate(rev_val, growth_mu, rev_sigma, ebitda_mu, margin_sigma, tax_rate, capex_val, wacc, years)

# --- 4. DASHBOARD UI ---
st.title("🏛️ Board-Level Investment Decision Suite")
st.markdown(f"**Strategic Assessment:** `{strategy}`")

# METRIC CARDS
mean_v = np.mean(npvs)
p90 = np.percentile(npvs, 10)
prob_success = (npvs > 0).sum() / 10000 * 100

c1, c2, c3 = st.columns(3)
with c1: st.markdown(f'<div class="metric-card"><p class="metric-label">Expected NPV</p><p class="metric-value">${mean_v:,.1f}M</p></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="metric-card"><p class="metric-label">P90 Safety Floor</p><p class="metric-value">${p90:,.1f}M</p></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="metric-card"><p class="metric-label">Success Probability</p><p class="metric-value">{prob_success:.1f}%</p></div>', unsafe_allow_html=True)

st.write("---")

# THE 4 PROFESSIONAL DECISION CHARTS
col1, col2 = st.columns(2)

with col1:
    # CHART 1: CUMULATIVE DISTRIBUTION (THE SUCCESS CURVE)
    sorted_npv = np.sort(npvs)
    y_vals = np.linspace(0, 100, len(sorted_npv))
    fig1 = px.line(x=sorted_npv, y=y_vals, title="1. Probability of Target Achievement",
                   labels={'x': 'Net Present Value ($M)', 'y': 'Confidence Level (%)'},
                   color_discrete_sequence=['#002D62'])
    fig1.add_vrect(x0=sorted_npv[0], x1=0, fillcolor="red", opacity=0.1, annotation_text="Loss Zone")
    st.plotly_chart(fig1, use_container_width=True)
    
    # CHART 2: SENSITIVITY TORNADO (KEY DRIVERS)
    st.write(" ")
    drivers = ['Growth Rate', 'EBITDA Margin', 'WACC', 'CapEx']
    # Calculating swing impact of +10% for each
    impacts = [
        np.mean(simulate(rev_val, growth_mu*1.1, rev_sigma, ebitda_mu, margin_sigma, tax_rate, capex_val, wacc, years, 1000)[0]) - mean_v,
        np.mean(simulate(rev_val, growth_mu, rev_sigma, ebitda_mu*1.1, margin_sigma, tax_rate, capex_val, wacc, years, 1000)[0]) - mean_v,
        np.mean(simulate(rev_val, growth_mu, rev_sigma, ebitda_mu, margin_sigma, tax_rate, capex_val, wacc*1.1, years, 1000)[0]) - mean_v,
        (mean_v - (mean_v - 3.5)) - mean_v # Fixed CapEx shift
    ]
    fig2 = px.bar(x=impacts, y=drivers, orientation='h', title="2. Tornado: What Moves the Needle?",
                  color=impacts, color_continuous_scale='RdBu_r', labels={'x': 'NPV Change ($M)', 'y': ''})
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    # CHART 3: STRATEGIC HEATMAP (THE DECISION MATRIX)
    w_range = np.linspace(wacc*0.7, wacc*1.3, 7)
    g_range = np.linspace(growth_mu*0.7, growth_mu*1.3, 7)
    # Heatmap math
    sens = [[(rev_val * (1+g) * ebitda_mu * (1-tax_rate)) / (w) - capex_val for w in w_range] for g in g_range]
    fig3 = px.imshow(sens, x=[f"{int(x*100)}% WACC" for x in w_range], y=[f"{int(y*100)}% Growth" for y in g_range],
                     text_auto='.1f', title="3. Matrix: Growth vs. Capital Cost", color_continuous_scale='RdYlGn')
    st.plotly_chart(fig3, use_container_width=True)

    # CHART 4: 5-YEAR UNCERTAINTY FAN (CASH FLOW RISK)
    years_list = [f"Year {i+1}" for i in range(years)]
    median_fcf = np.median(yearly_fcf, axis=0)
    p10_fcf = np.percentile(yearly_fcf, 10, axis=0)
    p90_fcf = np.percentile(yearly_fcf, 90, axis=0)
    
    fig4 = go.Figure([
        go.Scatter(x=years_list, y=p90_fcf, line=dict(width=0), showlegend=False),
        go.Scatter(x=years_list, y=p10_fcf, line=dict(width=0), fill='tonexty', fillcolor='rgba(0,45,98,0.2)', name='80% Confidence Band'),
        go.Scatter(x=years_list, y=median_fcf, line=dict(color='#002D62', width=3), name='Expected Cash Flow')
    ])
    fig4.update_layout(title="4. Forecast Uncertainty (5-Year Fan)", yaxis_title="FCF ($M)")
    st.plotly_chart(fig4, use_container_width=True)
