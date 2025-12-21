import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="Strategic Decision Intelligence", layout="wide")

# --- HIGH-END DASHBOARD CSS ---
st.markdown("""
<style>
    /* Main Background */
    .main { background-color: #f8f9fb; }
    
    /* Executive Metric Cards */
    .kpi-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #e1e4e8;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .kpi-label { font-size: 12px; color: #6a737d; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; }
    .kpi-value { font-size: 28px; color: #0366d6; font-weight: 800; margin: 5px 0; }
    
    /* Strategic Advice Cards with Custom Borders */
    .decision-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 4px;
        border-left: 6px solid #0366d6;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .decision-header { font-size: 18px; font-weight: 800; color: #1b1f23; margin-bottom: 10px; }
    .decision-body { font-size: 15px; color: #444d56; line-height: 1.6; }
    .highlight-green { color: #28a745; font-weight: 700; }
    .highlight-red { color: #d73a49; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# --- 1. DATA LAYER ---
@st.cache_data
def get_historical_data():
    dates = pd.date_range(end=datetime.today(), periods=48, freq='M')
    np.random.seed(42)
    return pd.DataFrame({
        'Month': dates, 
        'Revenue_M': np.linspace(10, 14, 48) * np.random.normal(1, 0.05, 48),
        'Margin': np.random.normal(0.35, 0.02, 48)
    })

hist_df = get_historical_data()
base_rev = hist_df.tail(12)['Revenue_M'].sum()
base_margin = hist_df.tail(12).Margin.mean()

# --- 2. SIDEBAR (KEPT PROFESSIONAL) ---
with st.sidebar:
    st.title("Boardroom Controls")
    strategy = st.selectbox("Strategic Option:", ["Organic Growth", "Premium Pricing", "Expansion", "Cost Optimization"])
    st.divider()
    
    with st.expander("Revenue & Growth Targets", expanded=True):
        rev_val = st.slider("Target Revenue ($M)", 50, 500, int(base_rev))
        growth_mu = st.slider("YoY Growth (%)", 0, 50, 10) / 100
        rev_sigma = st.slider("Revenue Volatility (%)", 5, 40, 20) / 100
        
    with st.expander("Operational Efficiency", expanded=True):
        ebitda_mu = st.slider("Target EBITDA Margin (%)", 10, 60, int(base_margin*100)) / 100
        margin_sigma = st.slider("Margin Volatility (%)", 5, 30, 12) / 100
        tax_rate = st.slider("Corporate Tax (%)", 15, 35, 25) / 100

    with st.expander("Capital & Market Risk", expanded=True):
        capex_val = st.slider("Initial Investment ($M)", 5, 100, 35)
        wacc = st.slider("Cost of Capital (WACC) (%)", 5, 20, 11) / 100
        years = 5

# --- 3. SIMULATION ENGINE ---
@st.cache_data
def run_simulation(rev, growth, r_vol, margin, m_vol, cpcl, disc, tax, yrs):
    np.random.seed(42)
    iters = 10000
    g_paths = np.random.normal(growth, r_vol, (iters, yrs))
    m_paths = np.random.normal(margin, m_vol, (iters, yrs))
    
    all_npvs = []
    for i in range(iters):
        revs = [rev]
        for y in range(yrs - 1): revs.append(revs[-1] * (1 + g_paths[i, y]))
        fcf = (np.array(revs) * m_paths[i]) * (1 - tax)
        npv = np.sum([fcf[t] / (1 + disc)**(t+1) for t in range(yrs)]) - cpcl
        all_npvs.append(npv)
    return np.array(all_npvs)

npvs = run_simulation(rev_val, growth_mu, rev_sigma, ebitda_mu, margin_sigma, capex_val, wacc, tax_rate, years)

# --- 4. EXECUTIVE UI ---
st.title("🏛️ Strategic Investment Analysis")

# Top KPI Row
success_rate = (npvs > 0).sum() / 10000 * 100
mean_npv = np.mean(npvs)
worst_case = np.percentile(npvs, 10)

kpi1, kpi2, kpi3 = st.columns(3)
with kpi1: st.markdown(f'<div class="kpi-card"><div class="kpi-label">Expected Net Value</div><div class="kpi-value">${mean_npv:,.1f}M</div></div>', unsafe_allow_html=True)
with kpi2: st.markdown(f'<div class="kpi-card"><div class="kpi-label">Confidence Score</div><div class="kpi-value">{success_rate:.1f}%</div></div>', unsafe_allow_html=True)
with kpi3: st.markdown(f'<div class="kpi-card"><div class="kpi-label">Conservative Floor</div><div class="kpi-value">${worst_case:,.1f}M</div></div>', unsafe_allow_html=True)

st.write(" ")

# Decision Grid
col_left, col_right = st.columns(2)

with col_left:
    # DESIGNED WRITING: DECISION 1
    st.markdown(f"""
    <div class="decision-card" style="border-left-color: #28a745;">
        <div class="decision-header">Decision 1: The Reliability Test</div>
        <div class="decision-body">
            Based on 10,000 market scenarios, this project has a <span class="highlight-green">{success_rate:.1f}%</span> 
            probability of profit. If the <span class="highlight-red">Red Zone</span> occupies more than 20% of the chart 
            below, the project requires immediate structural changes to <b>Margins</b> or <b>Initial CapEx</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    fig1 = px.histogram(npvs, nbins=60, color=(npvs > 0), 
                        color_discrete_map={True: '#28a745', False: '#d73a49'})
    fig1.update_layout(showlegend=False, margin=dict(t=0, b=0), height=350, 
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      xaxis_title="Potential Project Value ($M)", yaxis_title="")
    st.plotly_chart(fig1, use_container_width=True)

with col_right:
    # DESIGNED WRITING: DECISION 2
    st.markdown("""
    <div class="decision-card" style="border-left-color: #f66a0a;">
        <div class="decision-header">Decision 2: Strategic Resilience</div>
        <div class="decision-body">
            This matrix evaluates the plan against <b>Economic Shifts</b>. Find the center setup. 
            If moving <b>Down</b> (lower growth) or <b>Right</b> (higher interest rates) turns the boxes 
            <span class="highlight-red">Red</span>, your strategy is fragile and lacks a safety buffer.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Advanced 2D Sensitivity Grid
    w_range = np.linspace(wacc*0.6, wacc*1.4, 7)
    g_range = np.linspace(growth_mu*0.6, growth_mu*1.4, 7)
    sens = [[(rev_val * (1+g) * ebitda_mu * (1-tax_rate)) / (w) - capex_val for w in w_range] for g in g_range]
    
    fig2 = px.imshow(sens, 
                     x=[f"{int(x*100)}% Risk" for x in w_range], 
                     y=[f"{int(y*100)}% Growth" for y in g_range],
                     color_continuous_scale='RdYlGn', text_auto='.1f')
    fig2.update_layout(margin=dict(t=0, b=0), height=350, paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig2, use_container_width=True)

# Executive Verdict Bar
st.write(" ")
if success_rate >= 80:
    st.success(f"**EXECUTIVE VERDICT:** The **{strategy}** path is highly resilient. Approved for implementation based on current risk parameters.")
else:
    st.error(f"**EXECUTIVE VERDICT:** Strategic risk exceeds 20% threshold. Recommendation: **Re-evaluate {strategy} model** or increase efficiency targets.")
