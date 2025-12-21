import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="Executive Decision Board", layout="wide")

# --- CUSTOM CSS (KEPT EXACTLY AS PROVIDED) ---
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
    margin_m = np.random.normal(0.35, 0.02, 48)
    return pd.DataFrame({'Month': dates, 'Revenue_M': rev_m, 'EBITDA_Margin': margin_m})

hist_df = get_historical_data()
base_rev = hist_df.tail(12)['Revenue_M'].sum()
base_margin = hist_df.tail(12)['EBITDA_Margin'].mean()

# --- 2. SIDEBAR (KEPT EXACTLY AS PROVIDED) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1541/1541415.png", width=60)
    st.title("Control Center")
    strategy = st.selectbox("Select Growth Lever:", ["Organic Growth", "Premium Pricing", "Aggressive Expansion", "Cost Optimization"])
    
    st.divider()
    with st.expander("Revenue & Growth", expanded=True):
        rev_val = st.slider("Target Revenue ($M)", 50, 500, int(base_rev))
        growth_mu = st.slider("Expected Growth (%)", 0, 50, 10) / 100
        rev_sigma = st.slider("Revenue Volatility (%)", 5, 40, 20) / 100
        
    with st.expander("Operations & Tax", expanded=True):
        ebitda_mu = st.slider("Target Margin (%)", 10, 60, int(base_margin*100)) / 100
        tax_rate = st.slider("Effective Tax Rate (%)", 15, 35, 25) / 100
        margin_sigma = st.slider("Margin Volatility (%)", 5, 30, 12) / 100

    with st.expander("Capital & Risk", expanded=True):
        capex_val = st.slider("Investment/CapEx ($M)", 5, 100, 35)
        wacc = st.slider("WACC / Discount Rate (%)", 5, 20, 11) / 100
        years = 5

# --- 3. SIMULATION ENGINE ---
@st.cache_data
def run_simulation(rev, growth, r_vol, margin, tax, cpcl, disc, yrs):
    np.random.seed(42)
    iters = 10000
    g_paths = np.random.normal(growth, r_vol, (iters, yrs))
    all_npvs = []
    for i in range(iters):
        revs = [rev]
        for y in range(yrs - 1): revs.append(revs[-1] * (1 + g_paths[i, y]))
        fcf = (np.array(revs) * margin) * (1 - tax)
        npv = np.sum([fcf[t] / (1 + disc)**(t+1) for t in range(yrs)]) - cpcl
        all_npvs.append(npv)
    return np.array(all_npvs)

npvs = run_simulation(rev_val, growth_mu, rev_sigma, ebitda_mu, tax_rate, capex_val, wacc, years)

# --- 4. DASHBOARD UI ---
st.title("📊 Strategic Business Simulator")
st.markdown(f"**Current Strategy:** `{strategy}` | **Baseline:** 4-Year Monthly Historical Trend")

# KPI CARDS
mean_profit = np.mean(npvs)
success_rate = (npvs > 0).sum() / 10000 * 100
safety_floor = np.percentile(npvs, 10)

c1, c2, c3 = st.columns(3)
with c1: st.markdown(f'<div class="metric-card"><p class="metric-label">Expected Profit</p><p class="metric-value">${mean_profit:,.1f}M</p></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="metric-card"><p class="metric-label">Success Chance</p><p class="metric-value">{success_rate:.1f}%</p></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="metric-card"><p class="metric-label">Worst Case (P10)</p><p class="metric-value">${safety_floor:,.1f}M</p></div>', unsafe_allow_html=True)

st.write("---")

# --- EXECUTIVE DECISION GRID ---
col_left, col_right = st.columns(2)

with col_left:
    # --- CHART 1: THE CONFIDENCE DIAL ---
    st.info("### 🟢 DECISION 1: SUCCESS PROBABILITY")
    st.markdown("**Look for:** A high percentage. If this needle is in the **Red**, the risk of losing money is too high for this strategy.")
    
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number", value = success_rate,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence in Project Success", 'font': {'size': 18}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#0047AB"},
            'steps': [
                {'range': [0, 50], 'color': "#FFCDD2"},
                {'range': [50, 80], 'color': "#FFF9C4"},
                {'range': [80, 100], 'color': "#C8E6C9"}],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': success_rate}
        }))
    fig_gauge.update_layout(height=350, margin=dict(t=50, b=0, l=20, r=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_right:
    # --- CHART 2: THE STRESS TEST GRID ---
    st.warning("### 🟡 DECISION 2: MARKET RESILIENCE")
    st.markdown("**Look for:** Green boxes. If moving **down** (less growth) turns the grid **Red**, your plan is fragile.")
    
    w_range = np.linspace(wacc*0.7, wacc*1.3, 5)
    g_range = np.linspace(growth_mu*0.7, growth_mu*1.3, 5)
    sens = [[(rev_val * (1+g) * ebitda_mu * (1-tax_rate)) / (w) - capex_val for w in w_range] for g in g_range]
    
    fig_matrix = px.imshow(
        sens, 
        x=[f"{int(x*100)}% Risk" for x in w_range], 
        y=[f"{int(y*100)}% Growth" for y in g_range],
        text_auto='.1f', 
        color_continuous_scale='RdYlGn', 
        aspect="auto",
        labels=dict(x="Economic Risk (WACC)", y="Market Growth")
    )
    fig_matrix.update_layout(height=350, margin=dict(t=30, b=0, l=20, r=20))
    st.plotly_chart(fig_matrix, use_container_width=True)

# --- AUTOMATIC VERDICT BOX ---
st.write(" ")
if success_rate >= 80 and safety_floor > 0:
    st.success("#### ✅ STRATEGIC VERDICT: STRONG GO-AHEAD")
    st.write("This strategy is resilient and highly likely to deliver positive returns even in poor market conditions.")
elif success_rate >= 60:
    st.warning("#### ⚠️ STRATEGIC VERDICT: PROCEED WITH CAUTION")
    st.write("The project is profitable on average, but a market downturn could lead to significant capital loss.")
else:
    st.error("#### ❌ STRATEGIC VERDICT: RE-EVALUATE PLAN")
    st.write("The risk of loss is currently outside acceptable corporate safety thresholds.")

# --- FOOTER ---
with st.expander("📂 View 48-Month Historical Reference"):
    st.line_chart(hist_df.set_index('Month')['Revenue_M'])
