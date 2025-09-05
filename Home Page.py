import streamlit as st

st.set_page_config(page_title="Home", layout="wide")

st.image("logo.png", width=800)
st.markdown(
    """
    <style>
    .big-font {
        font-size:30px !important;
    }
    </style>
    """,
    unsafe_allow_html=True)

st.markdown(
    """
    <div class="big-font">
    The <b>Fixed Income Lab</b> is an interactive tool designed to showcase key concepts and analytical techniques surrounding fixed income securities and credit markets. It currently offers the following pages:

    <ul>
      <li>Bond Pricing and Risk Analysis</li>
      <li>Swaption Pricing using the Black Model</li>
      <li>Binomial Tree Interest Rate Modelling</li>
    </ul>

    Each page includes <b>interactive calculators, visualisations</b> and <b>model explanations</b> to enhance understanding.
    </div>
    """,
    unsafe_allow_html=True)
st.markdown("***")
st.header("Bond Pricing and Risk Analysis:")
st.markdown("""
    <div class="big-font">
    This page provides an interactive environment for valuing both coupon and zero-coupon bonds, with solvers available for yield and price depending on the input. Alongside valuation, the page reports several measures of interest rate sensitivity, including different versions of duration as well as convexity. For added visualisation, the page plots a comparison between the true price response to yield changes and the approximations obtained through first- and second-order adjustments. This allows users to see directly how higher-order terms improve the accuracy of risk measures when modelling bond price behaviour.
    </div>
    """, unsafe_allow_html=True)
st.markdown("\n")
st.markdown("\n")
st.header("Swaption Pricing using the Black Model:")
st.markdown("""
    <div class="big-font">
    This page contains a customisable pricing engine for swaptions, making use of Black's model. Alongside price, the engine reports the swaption’s key Greeks, giving insight into how the value responds to changes in underlying parameters. Beyond standard sensitivities, the tool also calculates PV01, a measure of the change in price resulting from a parallel shift in the yield curve. To make this more informative, the PV01 is isolated to each swap payment maturity, clearly identifying the maturities to which the swaption is most sensitive.
    </div>
    """, unsafe_allow_html=True)
st.markdown("\n")
st.markdown("\n")
st.header("Binomial Tree Interest Rate Modelling:")
st.markdown("""
    <div class="big-font">
    This page performs calibration of both the Ho-Lee and Simple Black-Derman-Toy short-rate models, either separately or side by side for comparison. The calibration can be performed using either bond yields or discount factors as inputs, giving flexibility in how market information is incorporated. Once calibrated, the resulting trees provide an exact fit of the observed term structure and illustrate how rates evolve under each model.
    </div>
    """, unsafe_allow_html=True)
st.markdown("***")
st.markdown("""
    <div class="big-font">
    In-depth explanations are available in the dropdown box at the bottom of each page.
    </div>
    """, unsafe_allow_html=True)
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
