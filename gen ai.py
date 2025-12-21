import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="Executive Decision Board", layout="wide")

# --- CUSTOM CSS FOR BORDERS & METRICS ---
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
</style>
""", unsafe_allow_html=True)

# --- 1. DATA LAYER (4-Year Monthly History) ---
@st.cache_data
def get_historical_data():
    dates = pd.date_range(end=datetime.today(), periods=48, freq='M')
    np.random.seed(42)
    # Simulating 48 months of data with a growth trend
    rev_m = np.linspace(10.0, 14.0, 48) * np.random.normal(1, 0.05, 48)
    margin_m = np.random.normal(0.35, 0.02, 48)
    return pd.DataFrame({'Month': dates, 'Revenue_M': rev_m, 'EBITDA_Margin': margin_m})

hist_df = get_historical_data()
base_rev = hist_df.tail(12)['Revenue_M'].sum()
base_margin = hist_df.tail(12)['EBITDA_Margin'].mean()

# --- 2. SIDEBAR CONTROLS (Professional Grouping) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1541/1541415.png", width=60)
    st.title("Strategic Settings")
    strategy = st.selectbox("Current Path:", ["Organic Growth", "Premium Pricing", "Aggressive Expansion", "Cost Optimization"])
    
    st.divider()
    with st.expander("📈 Revenue & Growth", expanded=True):
        rev_val = st.slider("Target Annual Revenue ($M)", 50, 500, int(base_rev))
        growth_mu = st.slider("Expected Yearly Growth (%)", 0, 50, 10) / 100
        rev_sigma = st.slider("Revenue Volatility (%)", 5, 40, 20) / 100
        
    with st.expander("💰 Profit & Risk", expanded=True):
        ebitda_mu = st.slider("Target Margin (%)", 10, 60, int(base_margin*100)) / 100
        wacc = st.slider("WACC (Market Risk) (%)", 5, 20, 11) / 100
        capex_val = st.slider("Total Investment ($M)", 5, 100, 35)
    
    tax_rate = 0.25
    years = 5

# --- 3. SIMULATION ENGINE (Monte Carlo 10K Runs) ---
@st.cache_data
def run_simulation(rev, growth, r_vol, margin, cpcl, disc, yrs):
    np.random.seed(42)
    iters = 10000
    g_paths = np.random.normal(growth, r_vol, (iters, yrs))
    all_npvs = []
    for i in range(iters):
        rev_path = [rev]
        for y in range(yrs - 1):
            rev_path.append(rev_path[-1] * (1 + g_paths[i, y]))
        fcf = (np.array(rev_path) * margin) * (1 - tax_rate)
        npv = np.sum([fcf[t] / (1 + disc)**(t+1) for t in range(yrs)]) - cpcl
        all_npvs.append(npv)
    return np.array(all_npvs)

npvs = run_simulation(rev_val, growth_mu, rev_sigma, ebitda_mu, capex_val, wacc, years)

# --- 4. MAIN DASHBOARD ---
st.title("📊 Strategic Investment Decision Board")
st.markdown(f"**Strategy Focus:** {strategy} | **Analysis Period:** 5 Years")

# TOP METRIC CARDS
mean_profit = np.mean(npvs)
success_rate = (npvs > 0).sum() / 10000 * 100
p10_safety = np.percentile(npvs, 10)

m1, m2, m3 = st.columns(3)
with m1: st.markdown(f'<div class="metric-card"><p class="metric-label">Expected Total Profit</p><p class="metric-value">${mean_profit:,.1f}M</p></div>', unsafe_allow_html=True)
with m2: st.markdown(f'<div class="metric-card"><p class="metric-label">Chance of Success</p><p class="metric-value">{success_rate:.1f}%</p></div>', unsafe_allow_html=True)
with m3: st.markdown(f'<div class="metric-card"><p class="metric-label">Safety Floor (Worst Case)</p><p class="metric-value">${p10_safety:,.1f}M</p></div>', unsafe_allow_html=True)

# --- AUTOMATIC STRATEGIC VERDICT ---
st.write(" ")
if success_rate >= 85:
    st.success("### ✅ STRATEGIC VERDICT: STRONG GO-AHEAD")
    st.info("The project shows high resilience. The probability of hitting profit targets is superior.")
elif success_rate >= 70:
    st.warning("### ⚠️ STRATEGIC VERDICT: PROCEED WITH CAUTION")
    st.info("The project is profitable but highly sensitive to market shifts. Monitor margins closely.")
else:
    st.error("### ❌ STRATEGIC VERDICT: RE-EVALUATE PLAN")
    st.info("The risk of capital loss is currently too high. Adjust the investment or increase targets.")

st.divider()

# --- 2-CHART DECISION GRID ---
col_left, col_right = st.columns(2)

with col_left:
    # 🟢 HIGHLIGHTED DECISION 1
    st.info("### 🟢 DECISION 1: THE CONFIDENCE TEST")
    st.markdown("""
    **Actionable Insight:** Is the **Green area** significantly larger than the **Red**?  
    > **Rule:** If the *Chance of Success* is below **80%**, the risk is high. Redesign the plan.
    """)
    
    fig1 = px.histogram(
        npvs, nbins=50, 
        color=(npvs > 0), 
        color_discrete_map={True: '#28a745', False: '#dc3545'},
        labels={'value': 'Total Profit/Loss ($M)', 'count': 'Scenarios'}
    )
    fig1.add_vline(x=0, line_width=2, line_color="black")
    fig1.update_layout(showlegend=False, margin=dict(t=10))
    st.plotly_chart(fig1, use_container_width=True)

with col_right:
    # 🟡 HIGHLIGHTED DECISION 2
    st.warning("### 🟡 DECISION 2: THE STRESS TEST")
    st.markdown("""
    **Actionable Insight:** Look at the center box. If moving **Down** (Less Growth) turns the boxes **Red**...  
    > **Rule:** Your plan is **too fragile**. Ensure the strategy works even if the market slows down.
    """)
    
    w_range = np.linspace(wacc*0.7, wacc*1.3, 5)
    g_range = np.linspace(growth_mu*0.7, growth_mu*1.3, 5)
    sens = [[(rev_val * (1+g) * ebitda_mu * (1-0.25)) / (w) - capex_val for w in w_range] for g in g_range]
    
    fig2 = px.imshow(
        sens, 
        x=[f"{int(x*100)}% Risk" for x in w_range], 
        y=[f"{int(y*100)}% Growth" for y in g_range],
        text_auto='.1f', 
        color_continuous_scale='RdYlGn', 
        aspect="auto"
    )
    fig2.update_layout(margin=dict(t=10))
    st.plotly_chart(fig2, use_container_width=True)

# --- 5. HISTORICAL REFERENCE ---
with st.expander("📂 View Historical 4-Year Performance Baseline"):
    st.line_chart(hist_df.set_index('Month')['Revenue_M'])
    st.dataframe(hist_df, use_container_width=True)
