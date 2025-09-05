import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm
import streamlit as st

def discount_factors(t, T, freq, reset=False):
    step = 1 if freq == "Annual" else 0.5
    times = np.arange(t, T + step, step)

    dfs = {}
    # chunk times into groups of 3 for nicer display
    for i in range(0, len(times), 3):
        cols = st.columns(3)
        for j, k in enumerate(times[i:i+3]):
            key = f"df_{k}"
            default_val = round(np.exp(-0.03*k), 2)
            if reset:
                st.session_state[key] = default_val
            init_val = st.session_state.get(key, default_val)
            with cols[j]:
                dfs[k] = st.number_input(
                    f"Year {k:.1f}",
                    min_value=0.02,
                    max_value=1.1,
                    value=init_val,
                    step=0.01,
                    key=key
                )
    return dfs

def black_price(N, t, T, K, sigma, type_, freq, dfs):
    step = 1 if freq == "Annual" else 0.5
    A0 = (sum(dfs.values()) - dfs[t]) * step
    if A0 <= 0:
        st.error("Annuity computed as zero or negative, check discount factors")
        return "Error"
    F = (dfs[t] - dfs[T]) / A0
    if F <= 0:
        st.error("Forward rate computed as zero or negative. The Black model assumes lognormal distribution of the forward rate.")
        return "Error"
    d1 = (np.log(F/K) + 0.5*sigma**2*t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    if type_ == "Payer":
        V = N * A0 * (F*Nd1 - K*Nd2)
        delta = N * A0 * Nd1 / 10000 # Scale per basis point
    elif type_ == "Receiver":
        V = N * A0 * (K*(1-Nd2) - F*(1-Nd1))
        delta = N * A0 * (Nd1-1) / 10000 # Scale per basis point
    gamma = (N * A0 * norm.pdf(d1) / (F * sigma * np.sqrt(t))) / 10000**2 # Scale per bp squared
    vega  = N * A0 * F * np.sqrt(t) * norm.pdf(d1) * 0.01 # Scale per ppt
    return V, delta, gamma, vega, F

def pv01(N, t, T, K, sigma, type_, freq, dfs, V):
    bp = 0.0001
    dfs_up = {}
    dfs_down = {}
    for year, df in dfs.items():
        r = -np.log(df) / year
        r_up = r + bp
        r_down = r - bp
        dfs_up[year] = np.exp(-r_up * year)
        dfs_down[year] = np.exp(-r_down * year)
    p_up, _, _, _, _ = black_price(N, t, T, K, sigma, type_, freq, dfs_up)
    p_down, _, _, _, _ = black_price(N, t, T, K, sigma, type_, freq, dfs_down)
    pv01 = p_up - p_down
    return pv01

def pv01_per_date(N, t, T, K, sigma, type_, freq, dfs, V):
    bp = 0.0001
    table = []
    for year, df in dfs.items():
        dfs_up = dfs.copy()
        dfs_down = dfs.copy()
        r = -np.log(df) / year
        r_up = r + bp
        r_down = r - bp
        dfs_up[year] = np.exp(-r_up * year)
        dfs_down[year] = np.exp(-r_down * year)
        p_up, _, _, _, _ = black_price(N, t, T, K, sigma, type_, freq, dfs_up)
        p_down, _, _, _, _ = black_price(N, t, T, K, sigma, type_, freq, dfs_down)
        pv01 = p_up - p_down
        table.append({"Year": year, "PV01": pv01})
    return pd.DataFrame(table)  


st.set_page_config(page_title="Swaption Pricing", layout="wide")

st.markdown(f"""
<div style="display:flex; align-items:center; gap:10px; margin-top:20px;">
    <a href="https://www.linkedin.com/in/e-chamberlain-hall/" target="_blank">
        <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" 
             width="30" height="30" alt="LinkedIn Logo">
    </a>
    <a href="https://www.linkedin.com/in/e-chamberlain-hall/" target="_blank" 
       style="color:#B390D4; font-size:16px; text-decoration:none;">
        LinkedIn
    </a>
</div>
""", unsafe_allow_html=True)

st.title("Black Model Swaption Pricing Engine")
st.markdown(
    """
    <style>
    /* Style the button itself */
    div.stButton > button:first-child {
        background-color: #3368A6;   /* custom background */
        color: white;                /* text color */
        font-weight: bold;           /* bold text */
        padding: 3px 8px;          /* bigger button */
        border-radius: 12px;         /* rounded corners */
        margin: auto;                /* center horizontally */
        display: block;              /* center as block element */
    }

    /* Style the text inside the button */
    div.stButton > button:first-child p {
        font-size: 20px;             /* bigger font size */
        font-weight: bold;           /* ensure boldness */
    }
    .btn-container {
        display: flex;
        justify-content: center;
        align-items: center;
    """, unsafe_allow_html=True)

warning = False
with st.container(border=True):
    st.subheader("Parameters")
    col1, col2 = st.columns(2)
    with col1:
        type_ = st.selectbox("Swaption Type", ("Payer", "Receiver"))
        freq = st.selectbox("Coupon Frequency", ("Annual", "Semi-Annual"))
        t = st.slider("Swaption Expiry", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
        sigma = st.slider("Volatility", min_value=0.0, max_value=1.0, value=0.2, step=0.001, format="%.3f")
    with col2:
        N = st.number_input("Notional", min_value=100, value=1_000_000, step=100_000)
        K = st.number_input("Strike Rate (%)", min_value=1.0, max_value=10.0, value=4.0, step=0.01, format="%.3f") / 100

        T = st.slider("Swap Maturity", min_value=0.5, max_value=10.0, value=5.0, step=0.5) 
    if T <= t:
        st.warning("Swap Maturity must be greater than Swaption Expiry")
        warning = True
    if (T-t) % 1 != 0 and freq == "Annual":
        st.warning("Swap Duration Must Match Coupon Frequency")
        warning = True
if not warning:
    with st.container(border=True):
        st.subheader("Discount Factors")
        reset_button = st.button("🔁 Reset")
        dfs = discount_factors(t, T, freq, reset=reset_button)
    output = black_price(N, t, T, K, sigma, type_, freq, dfs)
    if output == "Error":
        st.markdown(f"""
            <div style="
                background-color:#890a1e;
                color:#fbfef4;
                padding:15px;
                border-radius:10px;
                font-size:38px;
                font-weight:bold;
                text-align:center;">
                Pricing Error
            </div>
            """, unsafe_allow_html=True)
    else:
        V, delta, gamma, vega, F = output
        st.subheader(f"Forward Rate: {F*100:.2f}%")
        pv01 = pv01(N, t, T, K, sigma, type_, freq, dfs, V)
        pv01_table = pv01_per_date(N, t, T, K, sigma, type_, freq, dfs, V)
        rpv01_table = round(pv01_table, 2)
        pv01_table.set_index("Year", inplace=True) # Set index after creating rounded table so it stays as a column to be plotted
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pv01_table.index, y=pv01_table["PV01"],
            mode="lines+markers",
            line=dict(width=6, color="#028cca"),
            marker=dict(size=14, color="#fb4f05", symbol="diamond")
        ))
        #fig.update_traces(line_color="#CFFF04")
        fig.update_layout(
            height = 800,
            title=dict(text="Plot of PV01 at each Maturity", font=dict(size=25)),
            xaxis=dict(title=dict(text="Maturity", font=dict(size=25)), tickfont=dict(size=25), dtick=0.5), 
            yaxis=dict(title=dict(text="PV01", font=dict(size=25)), tickfont=dict(size=25))
        )
        
        fig1 = go.Figure(data=[go.Table(
            header=dict(values=list(rpv01_table.columns),
                        align='center',
                        font=dict(size=20, weight="bold"),
                        height=40),
            cells=dict(values=[rpv01_table[col] for col in rpv01_table.columns],
                       align='center',
                       font=dict(size=20),
                       height=40))
        ])
        fig1.update_layout(
            height=81*T,
            margin=dict(l=5, r=5, b=0, t=2)
           )
        st.markdown(f"""
            <div style="
                background-color:#38830e;
                color:#fbfef4;
                padding:15px;
                border-radius:10px;
                font-size:38px;
                font-weight:bold;
                text-align:center;">
                Price: £{V:,.2f}
            </div>
            """, unsafe_allow_html=True)
        st.markdown("***") # Horizontal line
        st.markdown(f"""
            <div style="display:flex; gap:10px;">
                <div style="
                    background-color:#44a4c4;
                    color:#fbfef4;
                    padding:15px;
                    border-radius:10px;
                    font-size:24px;
                    font-weight:bold;
                    text-align:center;
                    flex:1;">
                    Delta: £{delta:,.2f}
                </div>
                <div style="
                    background-color:#f39c12;
                    color:#fbfef4;
                    padding:15px;
                    border-radius:10px;
                    font-size:24px;
                    font-weight:bold;
                    text-align:center;
                    flex:1;">
                    Gamma: £{gamma:,.2f}
                </div>
                <div style="
                    background-color:#872657;
                    color:#fbfef4;
                    padding:15px;
                    border-radius:10px;
                    font-size:24px;
                    font-weight:bold;
                    text-align:center;
                    flex:1;">
                    Vega: £{vega:,.2f}
                </div>
                <div style="
                    background-color:#1252cf;
                    color:#fbfef4;
                    padding:15px;
                    border-radius:10px;
                    font-size:24px;
                    font-weight:bold;
                    text-align:center;
                    flex:1;">
                    PV01: £{pv01:,.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
            <div style="text-align:center; font-size:30px; font-weight:bold;">
                PV01 at each Maturity
            </div>
        """, unsafe_allow_html=True)
        st.markdown("\n") # Line break
        st.plotly_chart(fig1)
        st.plotly_chart(fig)

st.markdown(
    """
    <style>
    .big-font {
        font-size:25px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
with st.expander("💡 Learn more about the Black Model & Swaption Sensitivities"):
    st.title("Explaining the Model:")
    st.markdown("""
    <div class="big-font">
    The <b>Black Swaption Model</b> is used to price European options on interest rate swaps. A <b>Payer Swaption</b> gives the holder the right, but not the obligation, to enter a swap at option expiry paying fixed at the strike rate and receiving at the floating rate, while a <b>Receiver Swaption</b> gives the right to receive fixed and pay floating. Swaptions can be used to guarantee payment equal to a given percentage of the notional. For example, if a receiver swaption ends in the money, the holder can simultaneously enter a payer swap of the same notional. The floating legs cancel out, leaving a net fixed payment that is equal to the difference in the two swap rates as a percentage of the notional.<br><br>
    Conceptually, the model treats the <b>Forward Swap Rate</b> as the underlying variable, assuming it follows a lognormal distribution. The swaption price is computed as the discounted expected payoff under the risk-neutral measure, using the <b>Annuity Factor</b> to convert swap rates into present value. The <b>Annuity Factor</b> is calculated as the present value of future fixed payments, based on the relevant discount factors for each coupon period. This annuity is then used to determine the <b>Forward Swap Rate</b>. Finally, these quantities are substituted into the standard Black option formula to determine the price. The model separately handles payer and receiver swaptions using the call/put analogy.
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r"""\Large
    \begin{aligned}
    A_0 = \sum_{i=1}^{N} DF(T_i)\times\Delta_i\;,&\hspace{50px}F = \frac{DF(t) - DF(T)}{A_0}\\
    d_1 = \frac{\ln(F/K) + \frac{1}{2} \sigma^2 t}{\sigma \sqrt{t}}\;,&\hspace{50px}d_2 = d_1 - \sigma \sqrt{t} \\
    V_{\text{Payer}} = N \times A_0 \times \left[ F \, N(d_1) - K \, N(d_2) \right]\;,&\hspace{50px}V_{\text{Receiver}} = N \times A_0 \times \left[ K \, N(-d_2) - F \, N(-d_1) \right]\\
    \end{aligned}
    """)
    st.markdown("""
    <div class="big-font">
    Here, <b>t</b> represents the expiry of the swaption and <b>T</b> represents the maturity of the swap (not to be confused with the tenor).
    </div>
    """, unsafe_allow_html=True)
    st.markdown("***")
    st.title("Swaption Sensitivities:")
    st.markdown("""
    <div class="big-font">
    The <b>Delta</b>, <b>Gamma</b> and <b>Vega</b> of the swaption measure its sensitivity to first & second order changes in the forward swap rate as well as to changes in volatility. They are calculated by taking the relevant derivatives of the pricing formula:
    </div>
    """, unsafe_allow_html=True)
    st.latex(r"""\Large
    \begin{align*}
    \Delta = \frac{\partial{V}}{\partial{F}}&=\begin{cases}N\times A_0\times N(d_1) & \text{Payer}\\-N\times A_0\times N(-d_1) & \text{Receiver}\end{cases}\\\\
    \Gamma = \frac{\partial{\Delta}}{\partial{F}}&=\frac{N\times A_0\times N(d_{1})}{F\times\sigma\sqrt{t}}\\\\
    \textit{Vega} = \frac{\partial{V}}{\partial{\sigma}}&=N\times A_0\times F\times\sqrt{t}\times N(d_{1})
    \end{align*}
    """)
    st.markdown("""
    <div class="big-font">
    These formulae can be scaled in order to make the output more intuitive: 
    <ul>
        <li><b>Delta</b> is scaled to reflect a 1 basis point change in the forward swap rate.
        <li><b>Gamma</b> is scaled to reflect a 1 bp<sup>2</sup> change in the forward swap rate, thus for a change of 1 bp, Delta changes by approximately Gamma.
        <li><b>Vega</b> is scaled to refelct a 1 percentage point change in volatility.
    </ul>
    <b>PV01</b> measures the change in the swaption’s value resulting from a parallel shift in the yield curve of 1 basis point. It is calculated as the difference in value between the swaption priced on a curve shifted upward by 1 bp and one shifted downward by 1 bp. The same procedure can be applied to shifts in specific segments of the curve, allowing identification of the maturities to which the swaption is most sensitive. Sensitivities are typically concentrated around the swaption’s expiry and the underlying swap’s maturity, as these have the greatest influence on the forward swap rate. Changes in the forward rate alter the comparison between the forward rate and the strike rate, which is fundamental to determining the swaption’s value.
    </div>
    """, unsafe_allow_html=True)
