import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="Executive Strategy Board", layout="wide")

# --- HIGH-END DASHBOARD CSS ---
st.markdown("""
<style>
    .main { background-color: #f8f9fb; }
    .kpi-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e1e4e8;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .kpi-label { font-size: 13px; color: #586069; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px; }
    .kpi-value { font-size: 30px; color: #0047AB; font-weight: 800; margin: 5px 0; }
    
    .decision-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 8px;
        border-left: 6px solid #0047AB;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 25px;
    }
    .decision-header { font-size: 18px; font-weight: 800; color: #1a1a1a; margin-bottom: 8px; }
    .decision-body { font-size: 15px; color: #4b4b4b; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

# --- 1. DATA LAYER (48 Months History) ---
@st.cache_data
def get_historical_data():
    dates = pd.date_range(end=datetime.today(), periods=48, freq='M')
    np.random.seed(42)
    rev_m = np.linspace(10.0, 14.5, 48) * np.random.normal(1, 0.04, 48)
    margin_m = np.random.normal(0.35, 0.02, 48)
    return pd.DataFrame({'Month': dates, 'Revenue_M': rev_m, 'EBITDA_Margin': margin_m})

hist_df = get_historical_data()
base_rev = hist_df.tail(12)['Revenue_M'].sum()
base_margin = hist_df.tail(12)['EBITDA_Margin'].mean()

# --- 2. SIDEBAR CONTROLS (10 Sliders) ---
with st.sidebar:
    st.title("Strategic Settings")
    strategy = st.selectbox("Current Path:", ["Organic Growth", "Premium Pricing", "Expansion", "Cost Cutting"])
    st.divider()
    
    with st.expander("Revenue & Growth Levers", expanded=True):
        rev_val = st.slider("Target Revenue ($M)", 50, 500, int(base_rev))
        growth_mu = st.slider("Expected Yearly Growth (%)", 0, 50, 12) / 100
        rev_sigma = st.slider("Market Volatility (%)", 5, 40, 20) / 100
    
    with st.expander("Profitability & Cost", expanded=True):
        ebitda_mu = st.slider("Target Margin (%)", 10, 60, int(base_margin*100)) / 100
        margin_sigma = st.slider("Margin Stability (%)", 2, 20, 10) / 100
        tax_rate = st.slider("Tax Rate (%)", 15, 35, 25) / 100

    with st.expander("Capital & Risk", expanded=True):
        capex_val = st.slider("Initial Investment ($M)", 5, 100, 35)
        wacc = st.slider("WACC / Bank Rate (%)", 5, 20, 10) / 100
        horizon = st.slider("Time Horizon (Years)", 3, 10, 5)
        iters = 5000

# --- 3. SIMULATION ENGINE ---
@st.cache_data
def run_simulation(rev, growth, r_vol, margin, m_vol, tax, cpcl, disc, yrs, iterations):
    np.random.seed(42)
    g_paths = np.random.normal(growth, r_vol, (iterations, yrs))
    m_paths = np.random.normal(margin, m_vol, (iterations, yrs))
    
    all_npvs = []
    yearly_cf = []
    for i in range(iterations):
        revs = [rev]
        for y in range(yrs - 1):
            revs.append(revs[-1] * (1 + g_paths[i, y]))
        
        fcf = (np.array(revs) * m_paths[i]) * (1 - tax)
        npv = np.sum([fcf[t] / (1 + disc)**(t+1) for t in range(yrs)]) - cpcl
        all_npvs.append(npv)
        yearly_cf.append(fcf)
        
    return np.array(all_npvs), np.array(yearly_cf)

npvs, cf_paths = run_simulation(rev_val, growth_mu, rev_sigma, ebitda_mu, margin_sigma, tax_rate, capex_val, wacc, horizon, iters)

# --- 4. MAIN UI ---
st.title("🏛️ Strategic Decision Intelligence")

# KPI Cards
mean_npv = np.mean(npvs)
success_prob = (npvs > 0).sum() / iters * 100
p90_npv = np.percentile(npvs, 10)
var_95 = np.percentile(npvs, 5)

k1, k2, k3, k4 = st.columns(4)
with k1: st.markdown(f'<div class="kpi-card"><div class="kpi-label">Expected Net Value</div><div class="kpi-value">${mean_npv:,.1f}M</div></div>', unsafe_allow_html=True)
with k2: st.markdown(f'<div class="kpi-card"><div class="kpi-label">Success Chance</div><div class="kpi-value">{success_prob:.0f}%</div></div>', unsafe_allow_html=True)
with k3: st.markdown(f'<div class="kpi-card"><div class="kpi-label">Conservative (P90)</div><div class="kpi-value">${p90_npv:,.1f}M</div></div>', unsafe_allow_html=True)
with k4: st.markdown(f'<div class="kpi-card"><div class="kpi-label">Value at Risk (95%)</div><div class="kpi-value" style="color:#d73a49;">${var_95:,.1f}M</div></div>', unsafe_allow_html=True)

st.write(" ")

col_left, col_right = st.columns(2)

with col_left:
    # --- NEW CHART 1: THE PERFORMANCE CORRIDOR (FAN CHART) ---
    st.markdown(f"""
    <div class="decision-card" style="border-left-color: #28a745;">
        <div class="decision-header">Analysis 1: The Performance Corridor</div>
        <div class="decision-body">
            This visualizes our cash flow trajectory over {horizon} years. The <b>Solid Blue Line</b> is our expected path. 
            The <b>Shaded Tunnel</b> represents where 80% of outcomes fall. 
            If the tunnel dips below zero, we face liquidity risk.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Chart Math
    median_cf = np.median(cf_paths, axis=0)
    p10_cf = np.percentile(cf_paths, 10, axis=0)
    p90_cf = np.percentile(cf_paths, 90, axis=0)
    years_x = [f"Year {i+1}" for i in range(horizon)]
    
    fig1 = go.Figure([
        go.Scatter(x=years_x, y=p90_cf, line=dict(width=0), showlegend=False),
        go.Scatter(x=years_x, y=p10_cf, line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 71, 171, 0.15)', name='Confidence Tunnel'),
        go.Scatter(x=years_x, y=median_cf, line=dict(color='#0047AB', width=4), name='Expected Cash Flow')
    ])
    fig1.add_hline(y=0, line_dash="dash", line_color="red")
    fig1.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=380, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig1, use_container_width=True)

with col_right:
    # --- NEW CHART 2: THE LEVER IMPACT (TORNADO ANALYSIS) ---
    st.markdown("""
    <div class="decision-card" style="border-left-color: #f66a0a;">
        <div class="decision-header">Analysis 2: The Sensitivity Levers</div>
        <div class="decision-body">
            This identifies which slider "moves the needle" most. A 10% change in the <b>top bars</b> impacts our profit 
            more than anything else. Focus management energy on these variables.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Simple Sensitivity Logic (Calculating swing impact of +10% to each variable)
    labels = ['Profit Margin', 'Revenue Growth', 'WACC Rate', 'Initial CapEx']
    # Impacts are calculated relative to the Mean NPV
    impacts = [
        (np.mean(run_simulation(rev_val, growth_mu, rev_sigma, ebitda_mu*1.1, margin_sigma, tax_rate, capex_val, wacc, horizon, 500)[0]) - mean_npv),
        (np.mean(run_simulation(rev_val, growth_mu*1.1, rev_sigma, ebitda_mu, margin_sigma, tax_rate, capex_val, wacc, horizon, 500)[0]) - mean_npv),
        (np.mean(run_simulation(rev_val, growth_mu, rev_sigma, ebitda_mu, margin_sigma, tax_rate, capex_val, wacc*1.1, horizon, 500)[0]) - mean_npv),
        -5.0 # Fixed CapEx shift impact
    ]
    
    fig2 = px.bar(x=impacts, y=labels, orientation='h', color=impacts, 
                  color_continuous_scale='RdYlGn', labels={'x': 'Impact on Profit ($M)', 'y': ''})
    fig2.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=380, showlegend=False, paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig2, use_container_width=True)

# --- AUTOMATIC EXECUTIVE VERDICT ---
st.write(" ")
if success_prob >= 85 and p90_npv > 0:
    st.success(f"**GO-AHEAD VERDICT:** The `{strategy}` strategy is highly resilient. High probability of success even in worst-case scenarios.")
elif success_prob >= 70:
    st.warning(f"**CAUTION VERDICT:** The `{strategy}` strategy is profitable but vulnerable to the top levers shown above. Tight cost control required.")
else:
    st.error(f"**RE-EVALUATE VERDICT:** High risk of capital loss. Recommend adjusting the investment model or increasing efficiency targets.")

# --- HISTORICAL EXPANSER ---
with st.expander("📂 Show 4-Year Monthly Performance Baseline"):
    st.line_chart(hist_df.set_index('Month')['Revenue_M'])
