import numpy as np
from scipy.optimize import newton
import plotly.graph_objects as go
import streamlit as st

def bond_price(T, coupon, freq, fv, ytm): # Vectorised for ease of plotting later
    y_arr = np.atleast_1d(ytm)
    if coupon == 0:
        dfs = (1+y_arr[:, None] / freq) **(-T*freq)
        prices = (fv * dfs).flatten()
    else:
        periods = np.arange(1, int(T * freq) + 1)
        coupon_pmt = (coupon * fv) / freq
        
        dfs = (1 + y_arr[:, None] / freq) ** (-periods[None, :])
        pv_coupons = np.sum(coupon_pmt * dfs, axis=1)
        pv_fv = fv * dfs[:, -1]
        prices = pv_coupons + pv_fv

    return prices[0] if np.isscalar(ytm) else prices

def bond_ytm(price, T, coupon, freq, fv, guess=0.05):
    f = lambda ytm: bond_price(T, coupon, freq, fv, ytm) - price
    ytm = newton(f, guess)
    return ytm

def duration(T, coupon, freq, fv, ytm):
    y_arr = np.atleast_1d(ytm)
    periods = np.arange(1, int(T * freq) + 1)
    coupon_pmt = (coupon * fv) / freq
    cash_flows = np.full(periods.shape, coupon_pmt, dtype=float)
    cash_flows[-1] += fv 

    dfs = (1 + y_arr[:, None] / freq) ** (-periods[None, :])
    pv_cash_flows = cash_flows[None, :] * dfs

    prices = bond_price(T, coupon, freq, fv, y_arr)
    weighted_times = (periods[None, :] * pv_cash_flows)
    Dmac = np.sum(weighted_times, axis=1) / prices / freq
    Dmod = Dmac / (1 + y_arr / freq)
    DV01 = Dmod * prices * 0.0001

    if np.isscalar(ytm):
        return float(Dmac[0]), float(Dmod[0]), float(DV01[0])
    
    return Dmac, Dmod, DV01

def convexity(T, coupon, freq, fv, ytm):
    y_arr = np.atleast_1d(ytm)
    periods = np.arange(1, int(T * freq) + 1)
    coupon_pmt = (coupon * fv) / freq
    cash_flows = np.full(periods.shape, coupon_pmt, dtype=float)
    cash_flows[-1] += fv

    dfs_conv = (1 + y_arr[:, None] / freq) ** (-(periods[None, :] + 2))
    pv_conv_terms = cash_flows[None, :] * dfs_conv

    convexity_num = np.sum((periods[None, :] * (periods[None, :] + 1)) * pv_conv_terms, axis=1)
    prices = bond_price(T, coupon, freq, fv, y_arr) 
    prices = np.atleast_1d(prices)
    convexity_years = convexity_num / prices / (freq**2)
    dollar_convexity = prices * convexity_years

    if np.isscalar(ytm):
        return float(convexity_years[0]), float(dollar_convexity[0])
    return convexity_years, dollar_convexity

def approx_analysis(T, coupon, freq, fv, y0, n_points=500):
    dy = 0.1
    y_min = max(y0 - dy, 1e-6)
    y_max = y0 + dy
    yields = np.linspace(y_min, y_max, n_points)
    
    P0 = float(bond_price(T, coupon, freq, fv, y0))
    _, Dmod0, _ = duration(T, coupon, freq, fv, y0)
    Convexity0, _ = convexity(T, coupon, freq, fv, y0)  # Convexity0 is years^2

    # True prices along the grid
    true_prices = bond_price(T, coupon, freq, fv, yields)

    # Approximations (use Δy = y - y0)
    dy_arr = yields - y0
    linear_approx = P0 - Dmod0 * P0 * dy_arr
    quadratic_approx = linear_approx + 0.5 * Convexity0 * P0 * (dy_arr ** 2)
    yields *= 100
    y0 *= 100

    fig = go.Figure()
    
    # True price line
    fig.add_trace(go.Scatter(
        x=yields, y=true_prices,
        mode='lines',
        name='True Price',
        line=dict(width=6)
    ))
    
    # Duration (linear) approx
    fig.add_trace(go.Scatter(
        x=yields, y=linear_approx,
        mode='lines',
        name='Linear Duration approx',
        line=dict(dash='dash', width=6)
    ))
    
    # Duration + Convexity approx
    fig.add_trace(go.Scatter(
        x=yields, y=quadratic_approx,
        mode='lines',
        name='Quadratic Convexity approx',
        line=dict(dash='dot', width=6)
    ))
    
    # Vertical line at y0
    fig.add_vline(
        x=y0,
        line=dict(color='white', dash='dot', width=5),
        annotation_text=f"Initial YTM = {y0:.3f}%",
        annotation_position="top"
    )
    
    # Base point (P0)
    fig.add_trace(go.Scatter(
        x=[y0], y=[P0],
        mode='markers',
        name='Initial Price',
        marker=dict(color='white', size=10)
    ))
    
    fig.update_layout(
        width=225,
        height=800,
        title=dict(text=f"Price approximations about Initial YTM = {y0:.3f}%", font=dict(size=25)),
        xaxis=dict(title=dict(text='Yield to Maturity (Annualised %)', font=dict(size=25)), tickfont=dict(size=25), dtick=1),
        yaxis=dict(title=dict(text='Price', font=dict(size=25)), tickfont=dict(size=25)),
        template='plotly_white',
        legend=dict(bordercolor="white", borderwidth=1, font=dict(size=25))
    )

    return fig

st.set_page_config(page_title="Bond Analysis", layout="wide")

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

st.title("Bond Analysis")

freq = 1
warning = False
with st.container(border=True):
    st.subheader("Parameters")
    fv = st.number_input("Face Value", min_value=100, value=100, step=100)
    T = st.slider("Maturity", min_value=0.5, max_value=30.0, value=5.0, step=0.5)
    coupon = st.number_input("Coupon (Annualised %)", min_value = 0.0, max_value = 20.0, format="%.3f") / 100
    if coupon != 0:
        type_ = st.selectbox("Coupon Frequency", ("Annual", "Semi-Annual"))
        freq = 2 if type_ == "Semi-Annual" else freq
        if T % 1 != 0 and type_ == "Annual":
            st.warning("Maturity Must Match Compounding Frequency")
            warning = True
            solve_for = False
    if not warning:
        solve_for = st.selectbox("Solve for:", ("Price", "Yield to Maturity"))
        if solve_for == "Price":
            ytm = st.number_input("Yield to Maturity (Annualised %)", value = 5.0, format="%.3f") / 100
            price = bond_price(T, coupon, freq, fv, ytm)       
        elif solve_for == "Yield to Maturity":
            price = st.number_input("Price", min_value = 10.0, value = 78.353, format="%.3f")
            ytm = bond_ytm(price, T, coupon, freq, fv)

if solve_for == "Price":
    st.markdown(f"""
        <div style="
            background-color:#38830e;
            color:#fbfef4;
            padding:15px;
            border-radius:10px;
            font-size:38px;
            font-weight:bold;
            text-align:center;">
            Price: £{price:,.2f}
        </div>
        """, unsafe_allow_html=True)
elif solve_for == "Yield to Maturity":
    st.markdown(f"""
        <div style="
            background-color:#38830e;
            color:#fbfef4;
            padding:15px;
            border-radius:10px;
            font-size:38px;
            font-weight:bold;
            text-align:center;">
            YTM: {ytm*100:.3f}%
        </div>
        """, unsafe_allow_html=True)

if not warning:
    Dmac, Dmod, DV01 = duration(T, coupon, freq, fv, ytm)
    convexity_years, dollar_convexity = convexity(T, coupon, freq, fv, ytm)
    st.markdown("***")
    st.markdown(f"""
            <div style="display:flex; gap:10px;">
                <div style="
                    background-color:#3cb3da;
                    color:#fbfef4;
                    padding:15px;
                    border-radius:10px;
                    font-size:24px;
                    font-weight:bold;
                    text-align:center;
                    flex:1;">
                    Macaulay Duration: {Dmac:,.2f}
                </div>
                <div style="
                    background-color:#1b77b6;
                    color:#fbfef4;
                    padding:15px;
                    border-radius:10px;
                    font-size:24px;
                    font-weight:bold;
                    text-align:center;
                    flex:1;">
                    Modified Duration: {Dmod:,.2f}%
                </div>
                <div style="
                    background-color:#1f4488;
                    color:#fbfef4;
                    padding:15px;
                    border-radius:10px;
                    font-size:24px;
                    font-weight:bold;
                    text-align:center;
                    flex:1;">
                    DV01: £{DV01:,.2f}
                </div>
                <div style="
                    background-color:#201f59;
                    color:#fbfef4;
                    padding:15px;
                    border-radius:10px;
                    font-size:24px;
                    font-weight:bold;
                    text-align:center;
                    flex:1;">
                    Convexity (Annualised): {convexity_years:,.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("\n")
    fig = approx_analysis(T, coupon, freq, fv, ytm, n_points=500)
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
with st.expander("💡 Learn more about Bond Analysis"):
    st.header("Price and Yield To Maturity:")
    st.markdown("""
        <div class="big-font">
        A simple formula links a bond's yield to its price:
        </div>
        """, unsafe_allow_html=True)
    st.latex(r"""\Large
        P(T)=\sum_{i=1}^{f\times T}\;\frac{c\times FV}{(1+\dfrac{y}{f})^{i}}\quad+\quad\frac{FV}{(1+\dfrac{y}{f})^{f\times T}}\;,\qquad f=\begin{cases}1 & \text{Annual}\\2 & \text{Semi-annual, with halved coupon rate }c\end{cases}
        """)
    st.markdown("""
        <div class="big-font">
        By definition, a bond's <b>Yield to Maturity</b> is the constant rate that equates the present value of future cashflows to the bond's price. If these cashflows and the YTM are known, discounting the cashflows according to the YTM gives the bond's price. To calculate YTM from price, a common approach is to employ numerical methods as a closed-form solution is not always available, particularly with a larger number of cashflows. In this case, a Newton-Raphson root-finding algorithm is used to converge on the YTM.<br><br>
        With this pricing formula, it is possible to calculate various measures of price sensitivity to YTM.
        </div>
        """, unsafe_allow_html=True)
    st.header("Macaulay Duration:")
    st.markdown("""
        <div class="big-font">
        <b>Macaulay Duration</b> is defined as the weighted average of payment dates from a bond's cashflows, with weights equal to the disounted value of each cashflow relative to the bond's current price. Measured in years, it provides an estimate of how long it takes to recover the bond's price*.
        </div>
        """, unsafe_allow_html=True)
    st.latex(r"""\Large
        D_{\text{Mac}}=\sum_{i=1}^{T}\frac{c\times FV\;/\;(1+y)^{T_{i}}}{P(T)}\times T_{i}\quad+\quad\frac{FV\;/\;(1+y)^{T}}{P(T)}\times T
        """)
    st.header("Modified Duration:")
    st.markdown("""
        <div class="big-font">
        <b>Modified Duration</b> adjusts <b>Macaulay Duration</b> to directly measure price sensitivity to small changes in yield. It approximates the negative of the percentage change in price for a percentage point change in yield.<br><br>
        </div>
        """, unsafe_allow_html=True)
    st.latex(r"""\Large
    D_{\text{Mod}}=\frac{D_{\text{Mac}}}{(1+y)}
    """)
    st.header("DV01:")
    st.markdown("""
        <div class="big-font">
        <b>DV01</b> measures the negative of the dollar change in bond price for a 1 basis point change in yield. It translates duration into currency terms, making it particularly useful for risk management.<br><br>
        </div>
        """, unsafe_allow_html=True)
    st.latex(r"""\Large
        \text{DV01}=\frac{D_{\text{Mod}}\times P(T)}{10,000}
        """)
    st.header("Convexity:")
    st.markdown("""
        <div class="big-font">
        <b>Convexity</b> measures the curvature in the relationship between bond prices and yields, capturing how duration itself reacts to changes in yield. <b>Convexity</b> is calculated by taking the second derivative of the price formula with respect to yield and can improve price change estimates for larger yield shifts. When dealing with other compounding frequencies, scaling by the square of the frequency converts the units to  <b>years²</b>, allowing comparison between e.g. annual and semi-annual payments.<br><br>
        </div>
        """, unsafe_allow_html=True)
    st.latex(r"""\Large
        C=\sum_{i=1}^{T}\frac{c\times FV\;/\;(1+y)^{-(T_{i}+2)}}{P(T)}\times T_{i}(T_{i}+1)\quad+\quad\frac{FV\;/\;(1+y)^{-(T+2)}}{P(T)}\times T(T+1)
        """)
    st.header("Approximating Price Changes:")
    st.markdown("""
        <div class="big-font">
        The above measures can be used to approximate price changes via a Taylor series expansion of the bond's price around an initial yield. This approximation takes the form:
        </div>
        """, unsafe_allow_html=True)
    st.latex(r"""\Large
        P(T;\;y)\approx P(T;\;y_{0})\;-\;D_{\text{Mod}}\cdot P(T;\;y_{0})\cdot(y - y_{0})\;+\;\frac{1}{2}C\cdot P(T;\;y_{0})\cdot(y - y_{0})^{2}
        """)
    st.markdown("""
        <div class="big-font">
        Retaining only the first-order term involving <b>Modified Duration</b> gives a <b>linear approximation</b> of bond price changes which works well for small changes in yield, but quickly loses accuracy further from the expansion point, particularly for longer dated bonds which have greater second-order sensitivities. Including the second-order term involving <b>Convexity</b> adds curvature to the approximation, greatly improving accuracy over larger yield changes by accounting for the bond's non-linear response to interest rates. In practice, the quadratic approximation is usually sufficient as adding higher-order terms provides little additional benefit while increasing computational complexity.
        </div>
        """, unsafe_allow_html=True)
    st.markdown("\n")
    st.write("*Only formulas for annual compounding are included past this point but they can easily be adapted to facilitate other frequencies")
