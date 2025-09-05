import numpy as np
import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from io import BytesIO
from scipy.optimize import root_scalar
import streamlit as st
import pathlib

def load_css(file_path):
    with open(file_path) as f:
        st.html(f"<style>{f.read()}</style>")
css_path = pathlib.Path("assets/styles.css")
load_css(css_path)

class HoLee:
    def __init__(self, input_prices: dict[float, float], sigma: float):
        self.maturities = np.array(list(input_prices.keys()))
        self.prices = np.array(list(input_prices.values()))
        self.thetas = np.zeros(len(self.maturities) - 1)
        self.delta = self.maturities[0]
        self.r0 = -np.log(self.prices[0]/100) / self.delta
        self.sigma = sigma

        try:
            self.solved_thetas = self.solve_thetas()
            self.plot_solved_tree(self.solved_thetas)
        except Exception as e:
            self.solved_thetas = e

    def build_ho_lee_tree(self, thetas):
        n = len(self.maturities)
        
        # Initialize tree
        r_tree = [[0] * (i + 1) for i in range(n)]

        r_tree[0][0] = self.r0  # Set roots of the tree

        # Build tree iteratively
        for i in range(n - 1):
            for j in range(i + 1):
                up_move = r_tree[i][j] + thetas[i] * self.delta + self.sigma * np.sqrt(self.delta) # Define up and down moves separately to avoid unnecessary recalculation
                down_move = r_tree[i][j] + thetas[i] * self.delta - self.sigma * np.sqrt(self.delta)
                r_tree[i + 1][j] = up_move  # Only define r[i+1][j] as an up move from r[i][j] and not a down move from r[i][j-1]
                r_tree[i + 1][j + 1] = down_move
                
        return r_tree 

    def bond_price_at_0(self, maturity, r_tree):
        price_tree = [[100] * (i + 1) for i in range(maturity + 1)] # Initialises all prices as 100 before working backwards to find the price at time 0

        for i in range(maturity - 1, -1, -1):
            for j in range(i + 1):
                price_tree[i][j] = np.exp(-r_tree[i][j] * self.delta) * 0.5 * (price_tree[i + 1][j] + price_tree[i + 1][j + 1])  # Discounting formula

        return price_tree[0][0]  # This returns the root of the tree i.e. price at time 0

    def objective(self, theta_i, i):
        self.thetas[i] = theta_i  # Set objective variable based on the theta we want to solve for
        r_tree = self.build_ho_lee_tree(self.thetas)  # Enable solver to recursively build new trees with updated guess
        p_guess = self.bond_price_at_0(i + 2, r_tree)  # Bond price at maturity i+2 since thetas[i] determines price of the bond maturing at [i+2]
        return p_guess - self.prices[i + 1]  # Difference from the actual price is the objective (since prices is a zero indexed array, prices[i+1] is the bond maturing at [i+2])

    def solve_thetas(self):
        for i in range(len(self.thetas)):
            sol = root_scalar(self.objective, args=(i,), bracket=[-1, 1], method='brentq')
            if sol.converged:
                self.thetas[i] = sol.root  # Dynamically update thetas as solved values are found
            else:
                raise ValueError(f"Root finding failed for thetas[{i}]")

        return self.thetas

    def plot_solved_tree(self, thetas):
        r_tree = self.build_ho_lee_tree(thetas)
        max_depth = len(r_tree)
        theta_labels = [f"{(theta*100):.2f}" for theta in thetas] + [""]  # Multiply thetas by 100 for readability
    
        max_rows = max(len(arr) for arr in r_tree)
    
        # Build rows
        table_data = []
        m_row, i_row, j_row = ["0"], [], []
        for i in range(max_rows):
            i_row.append(i)
            j_row.append("")
            m_row.append(self.maturities[i])
        table_data.append(m_row)
        table_data.append(theta_labels)
        table_data.append(i_row)
        table_data.append(j_row)
    
        for i in range(max_rows):
            row = []
            for j in range(max_depth):
                if i < len(r_tree[j]):
                    row.append(f"{(r_tree[j][i]*100):.2f}%")
                else:
                    row.append("")
            table_data.append(row)
    
        # Row labels
        row_labels = ["Maturity", "θ (x 100)", "i", "j"] + [f"{i}" for i in range(max_rows)]
    
        # Bold the theta row using HTML
        table_data[1] = [f"<b>{val}</b>" for val in table_data[1]]
        row_labels[1] = "<b>θ (x 100)</b>"
    
        # Combine row labels + data
        all_cells = []
        for label, row in zip(row_labels, table_data):
            all_cells.append([label] + row)
    
        # Transpose into columns (Plotly expects per-column lists)
        col_data = list(map(list, zip(*all_cells)))
    
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=[""] * (max_depth + 1),
                align="center",
                line_color="white",
                fill_color="white",
                font=dict(size=25, color="black")
            ),
            cells=dict(
                values=col_data,
                align="center",
                line_color="white",
                fill_color="white",
                font=dict(size=20, color="black"),
                height=40,
                format=[None] * (max_depth + 1),
                suffix=[""] * (max_depth + 1),
            )
        )])
    
        # Add a little extra height buffer to avoid cutting the last row
        fig.update_layout(
            width=1200,
            height=(max_rows + 6) * 50,  # + buffer
            margin=dict(l=10, r=10, t=10, b=5)
        )
    
        return fig

class BDT:
    def __init__(self, input_prices: dict[float, float], sigma: float):
        self.maturities = np.array(list(input_prices.keys()))
        self.prices = np.array(list(input_prices.values()))
        self.thetas = np.zeros(len(self.maturities) - 1)
        self.delta = self.maturities[0]
        self.z0 = math.log(-math.log(self.prices[0]/100) / self.delta)
        self.sigma = sigma

        try:
            self.solved_thetas = self.solve_thetas()
            self.plot_solved_tree(self.solved_thetas)
        except Exception as e:
            self.solved_thetas = e

    def build_bdt_tree(self, thetas):
        n = len(self.maturities)
        
        # Initialize trees
        z_tree = [[0] * (i + 1) for i in range(n)]
        r_tree = [[0] * (i + 1) for i in range(n)]

        z_tree[0][0] = self.z0  # Set roots of the trees
        r_tree[0][0] = np.exp(z_tree[0][0])

        # Build tree iteratively
        for i in range(n - 1):
            for j in range(i + 1):
                up_move = z_tree[i][j] + thetas[i] * self.delta + self.sigma * np.sqrt(self.delta) # Define up and down moves separately to avoid calculation errors
                down_move = z_tree[i][j] + thetas[i] * self.delta - self.sigma * np.sqrt(self.delta)
                z_tree[i + 1][j] = up_move  # Only define z[i+1][j] as an up move from z[i][j] and not a down move from z[i][j-1]
                z_tree[i + 1][j + 1] = down_move

            for j in range(i + 2):
                r_tree[i + 1][j] = np.exp(z_tree[i + 1][j])

        return r_tree  # We only need the r tree so no need to store the z tree

    def bond_price_at_0(self, maturity, r_tree):
        price_tree = [[100] * (i + 1) for i in range(maturity + 1)] # Initialises all prices as 100

        # Work backwards to find price at time 0
        for i in range(maturity - 1, -1, -1):
            for j in range(i + 1):
                price_tree[i][j] = math.exp(-r_tree[i][j] * self.delta) * 0.5 * (price_tree[i + 1][j] + price_tree[i + 1][j + 1])  # Discounting formula

        return price_tree[0][0]  # Return price at time 0

    def objective(self, theta_i, i):
        self.thetas[i] = theta_i  # Set objective variable based on the theta we want to solve for
        r_tree = self.build_bdt_tree(self.thetas)  # Enable solver to recursively build new trees with updated guess
        p_guess = self.bond_price_at_0(i + 2, r_tree)  # Bond price at maturity i+2 since thetas[i] determines price of the bond maturing at [i+2]
        return p_guess - self.prices[i + 1]  # Difference from the actual price is the objective (thetas[i] determines p[i+1] which is the bond maturing at [i+2], since we take the first price as p0)

    def solve_thetas(self):
        for i in range(len(self.thetas)):
            sol = root_scalar(self.objective, args=(i,), bracket=[-1, 1], method='brentq')
            if sol.converged:
                self.thetas[i] = sol.root  # Dynamically update thetas as solved values are found
            else:
                raise ValueError(f"Root finding failed for thetas[{i}]")

        return self.thetas

    def plot_solved_tree(self, thetas):
        r_tree = self.build_bdt_tree(thetas)
        max_depth = len(r_tree)
        theta_labels = [f"{(theta*100):.2f}" for theta in thetas] + [""]  # Multiply thetas by 100 for readability
    
        max_rows = max(len(arr) for arr in r_tree)
    
        # Build rows
        table_data = []
        m_row, i_row, j_row = ["0"], [], []
        for i in range(max_rows):
            i_row.append(i)
            j_row.append("")
            m_row.append(self.maturities[i])
        table_data.append(m_row)
        table_data.append(theta_labels)
        table_data.append(i_row)
        table_data.append(j_row)
    
        for i in range(max_rows):
            row = []
            for j in range(max_depth):
                if i < len(r_tree[j]):
                    row.append(f"{(r_tree[j][i]*100):.2f}%")
                else:
                    row.append("")
            table_data.append(row)
    
        # Row labels
        row_labels = ["Maturity", "θ (x 100)", "i", "j"] + [f"{i}" for i in range(max_rows)]
    
        # Bold the theta row using HTML
        table_data[1] = [f"<b>{val}</b>" for val in table_data[1]]
        row_labels[1] = "<b>θ (x 100)</b>"
    
        # Combine row labels + data
        all_cells = []
        for label, row in zip(row_labels, table_data):
            all_cells.append([label] + row)
    
        # Transpose into columns (Plotly expects per-column lists)
        col_data = list(map(list, zip(*all_cells)))
    
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=[""] * (max_depth + 1),
                align="center",
                line_color="white",
                fill_color="white",
                font=dict(size=25, color="black")
            ),
            cells=dict(
                values=col_data,
                align="center",
                line_color="white",
                fill_color="white",
                font=dict(size=20, color="black"),
                height=40,
                format=[None] * (max_depth + 1),
                suffix=[""] * (max_depth + 1),
            )
        )])
    
        # Add a little extra height buffer to avoid cutting the last row
        fig.update_layout(
            width=1200,
            height=(max_rows + 6) * 50,  # + buffer
            margin=dict(l=10, r=10, t=10, b=20)
        )
    
        return fig

def zcb_prices(t, T, freq, reset=False):
    step = 1 if freq == "Annual" else 0.5
    times = np.arange(t+step, T + step, step)

    zcbs = {}
    # chunk times into groups of 3
    for i in range(0, len(times), 3):
        cols = st.columns(3)
        for j, k in enumerate(times[i:i+3]):
            key = f"df_{k}"
            default_val = round(np.exp(-0.03*k)*100, 4)
            if reset:
                st.session_state[key] = default_val
            init_val = st.session_state.get(key, default_val)
            with cols[j]:
                zcbs[k] = st.number_input(
                    f"Year {k:.1f}",
                    min_value=0.0,
                    max_value=120.0,
                    value=init_val,
                    step=0.01,
                    key=key
                )
    return zcbs

def zcb_from_yields(t, T, freq, reset=False):
    step = 1 if freq == "Annual" else 0.5
    times = np.arange(t+step, T + step, step)

    yields = {}
    zcbs = {}
    # chunk times into groups of 3
    for i in range(0, len(times), 3):
        cols = st.columns(3)
        for j, k in enumerate(times[i:i+3]):
            key = f"yield_{k}"
            default_val = round(3+0.5*np.log(k+1), 4)
            if reset:
                st.session_state[key] = default_val
            init_val = st.session_state.get(key, default_val)    
            with cols[j]:
                yields[k] = st.number_input(
                    f"Year {k:.1f}", min_value=0.0001, max_value=20.0, value=init_val, step=0.01, format="%.4f",
                    key=key
                ) / 100
    for m, y in yields.items():
        zcbs[m] = 100 * np.exp(-(y) * m)
    return zcbs


st.set_page_config(page_title="Interest Rate Trees", layout="wide")

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

st.markdown(
    """
    <style>
    .btn-container {
        display: flex;
        justify-content: center;
        align-items: center;
    }

    /* Unsolved Bubble */
    .grey_bubble {
        background-color: #A7A9B4;
        color: white;
        font-weight: bold;
        font-size: 52px;
        padding: 15px 40px;
        border-radius: 12px;
        display: inline-block;
    }

    /* Solved Bubble */
    .green_bubble {
        background-color: #4B864C;
        color: white;
        font-weight: bold;
        font-size: 52px;
        padding: 15px 40px;
        border-radius: 12px;
        display: inline-block;
    }
    /* Error Bubble */
    .yellow_bubble {
        background-color: #EE9B0F;
        color: white;
        font-weight: bold;
        font-size: 52px;
        padding: 15px 40px;
        border-radius: 12px;
        display: inline-block;
    }    
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Binomial Interest Rate Trees")
vols = {"Ho-Lee": 0.015, "Simple Black-Derman-Toy": 0.2}
model = st.selectbox("Select Model", ("Ho-Lee", "Simple Black-Derman-Toy", "Both"))
if model != "Both":
    st.header(model+" Model")
    dual = False
else:
    dual = True

warning = False
if dual == False:
    with st.container(border=True):
        freq = st.selectbox("Time Step Frequency", ("Annual", "Semi-Annual"))
        T = st.slider("Final Maturity", min_value=0.5, max_value=11.0, value=5.0, step=0.5)
        type_ = st.selectbox("Calibration Method", ("Yields", "ZCB Prices"))
        if T%1 != 0 and freq == "Annual":
            st.warning("Maturity Must Match Time Step Frequency")
            warning = True
        if not warning:
            if type_ == "ZCB Prices":
                reset_button = st.button("🔁 Reset", key="reset")
                zcbs = zcb_prices(0, T, freq, reset=reset_button)
            if type_ == "Yields":
                reset_button = st.button("🔁 Reset", key="reset")
                zcbs = zcb_from_yields(0, T, freq, reset=reset_button)
            sigma = st.number_input("Volatility", min_value=0.0, max_value=1.0, value=vols[model], step=0.01, format="%.4f")


if model == "Simple Black-Derman-Toy" and not warning:
    col1, col2 = st.columns([1,1], gap="small")
    with col1:
        solve = st.button("🚨 Solve Tree", key="solve")
        plot = False
    if solve:
        bdt_model = BDT(zcbs, sigma)
        if isinstance(bdt_model.solved_thetas, ValueError):
            st.header(f"Calibration failed - solver could not find a route. Try adjusting {type_} or press Reset")
            with col2:
                st.markdown(
                    '''
                    <div style="text-align: right;">
                        <div class="yellow_bubble">🚧 Model Error</div>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )
        else:
            solved_thetas = bdt_model.solve_thetas()
            solved_r_tree = bdt_model.build_bdt_tree(solved_thetas)
            fig = bdt_model.plot_solved_tree(solved_thetas)
            plot = True
            with col2:
                st.markdown(
                    '''
                    <div style="text-align: right;">
                        <div class="green_bubble">🌳 Tree Solved</div>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )
    else:
        with col2:
            st.markdown(
                '''
                <div style="text-align: right;">
                    <div class="grey_bubble">⏰ Awaiting Input</div>
                </div>
                ''',
                unsafe_allow_html=True
            )
    if plot==True:
        st.plotly_chart(fig)

elif model == "Ho-Lee" and not warning:
    col1, col2 = st.columns([1,1], gap="small")
    with col1:
        solve = st.button("🚨 Solve Tree", key="solve")
        plot = False
    if solve:
        hl_model = HoLee(zcbs, sigma)
        if isinstance(hl_model.solved_thetas, ValueError):
            st.header(f"Calibration failed - solver could not find a route. Try adjusting {type_} or press Reset")
            with col2:
                st.markdown(
                '''
                <div style="text-align: right;">
                    <div class="yellow_bubble">🚧 Model Error</div>
                </div>
                ''',
                unsafe_allow_html=True
            )
        else:
            solved_thetas = hl_model.solve_thetas()
            solved_r_tree = hl_model.build_ho_lee_tree(solved_thetas)
            fig = hl_model.plot_solved_tree(solved_thetas)
            plot = True
            with col2:
                st.markdown(
                    '''
                    <div style="text-align: right;">
                        <div class="green_bubble">🌳 Tree Solved</div>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )
    else:
        with col2:
            st.markdown(
                '''
                <div style="text-align: right;">
                    <div class="grey_bubble">⏰ Awaiting Input</div>
                </div>
                ''',
                unsafe_allow_html=True
            )
    if plot==True:
        st.plotly_chart(fig)

else:
    st.header("Ho Lee Model & Simple Black-Derman-Toy Model")
    with st.container(border=True):
        freq = st.selectbox("Time Step Frequency", ("Annual", "Semi-Annual"))
        T = st.slider("Final Maturity", min_value=0.5, max_value=10.0, value=5.0, step=0.5)
        type_ = st.selectbox("Calibration Method", ("Yields", "ZCB Prices"))
        if T%1 != 0 and freq == "Annual":
            st.warning("Maturity Must Match Time Step Frequency")
            warning = True
        if not warning:
            if type_ == "ZCB Prices":
                reset_button = st.button("🔁 Reset", key="reset")
                zcbs = zcb_prices(0, T, freq, reset=reset_button)
            if type_ == "Yields":
                reset_button = st.button("🔁 Reset", key="reset")
                zcbs = zcb_from_yields(0, T, freq, reset=reset_button)
            sigma_hl = st.number_input("Ho-Lee Volatility", min_value=0.0, max_value=1.0, value=vols["Ho-Lee"], step=0.01, format="%.4f")
            sigma_bdt = st.number_input("BDT Volatility", min_value=0.0, max_value=1.0, value=vols["Simple Black-Derman-Toy"], step=0.01, format="%.4f")
    col1, col2 = st.columns([1,1], gap="small")
    if not warning:
        with col1:
            solve = st.button("🚨 Solve Trees", key="solve")
            plot=False
        if solve:
            hl_model = HoLee(zcbs, sigma_hl)
            bdt_model = BDT(zcbs, sigma_bdt)
            if isinstance(hl_model.solved_thetas, ValueError):
                st.header(f"Calibration failed for Ho-Lee Model - solver could not find a route. Try adjusting {type_} or press Reset")
            elif isinstance(bdt_model.solved_thetas, ValueError):
                st.header(f"Calibration failed for BDT Model - solver could not find a route. Try adjusting {type_} or press Reset")
                with col2:
                    st.markdown(
                    '''
                    <div style="text-align: right;">
                        <div class="yellow_bubble">🚧 Model Error</div>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )
            else:
                solved_hl_thetas = hl_model.solve_thetas()
                solved_bdt_thetas = bdt_model.solve_thetas()
                solved_hl_r_tree = hl_model.build_ho_lee_tree(solved_hl_thetas)
                solved_bdt_r_tree = bdt_model.build_bdt_tree(solved_bdt_thetas)
                fig_hl = hl_model.plot_solved_tree(solved_hl_thetas)
                fig_bdt = bdt_model.plot_solved_tree(solved_bdt_thetas)
                plot=True
                with col2:
                    st.markdown(
                        '''
                        <div style="text-align: right;">
                            <div class="green_bubble">🌳 Trees Solved</div>
                        </div>
                        ''',
                        unsafe_allow_html=True
                    )
        else:
            with col2:
                st.markdown(
                    '''
                    <div style="text-align: right;">
                        <div class="grey_bubble">⏰ Awaiting Input</div>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )
        if plot==True:
            st.header("Ho-Lee Model:")
            st.plotly_chart(fig_hl)
            st.header("Simple Black-Derman-Toy Model:")
            st.plotly_chart(fig_bdt)

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

with st.expander("💡 Learn more about Binomial Interest Rate Trees"):
    st.markdown("""
    <div class="big-font">
    Both the <b>Ho–Lee Model</b> and the <b>Simple Black–Derman–Toy Model</b> belong to the family of short-rate interest rate models. These models focus on the evolution of the instantaneous short-term interest rate over time, rather than  directly modelling bond prices or yields.<br><br>
    They are built on the idea of a binomial interest rate tree, where at each time step the short rate can move up or down with equal probability. A key feature of both models is the inclusion of the θ (theta) parameter, which acts as an adjustment term to ensure that the model is <b>exactly calibrated</b> to the current term structure of interest rates. In other words, bond prices implied by a calibrated tree are consistent with the observed market yield curve.<br><br>
    The two models are defined by similar equations, but differ in that the Ho-Lee Model assumes a normal distribution of the short-rate thus allowing negative rates, while the BDT Model assumes a lognormal distribution, ensuring rates are always positive. These equations govern how the short-rate evolves at each node (i, j) of the tree, either moving up to (i+1, j) or down to (i+1, j+1).
    </div>
    """, unsafe_allow_html=True)
    
    st.header("Ho Lee Model:")
    st.latex(r"""\Large
    \begin{align*}
    r_{i+1,\;j}&=r_{i,\;j}+\theta_{i}\times\Delta+\sigma_{HL}\times\sqrt{\Delta}\\
    r_{i+1,\;j+1}&=r_{i,\;j}+\theta_{i}\times\Delta-\sigma_{HL}\times\sqrt{\Delta}
    \end{align*}
    """)
    st.header("Simple Black-Derman-Toy Model:")
    st.latex(r"""\Large
    \begin{align*}
    z_{i,\;j}&=ln(r_{i,\;j})\\
    z_{i+1,\;j}&=z_{i,\;j}+\theta_{i}\times\Delta+\sigma_{BDT}\times\sqrt{\Delta}\\
    z_{i+1,\;j+1}&=z_{i,\;j}+\theta_{i}\times\Delta-\sigma_{BDT}\times\sqrt{\Delta}
    \end{align*}
    """)
    st.markdown("\n")
    st.markdown("""
    <div class="big-font">
    It is important to note that the volatility parameter in the Ho-Lee Model is the volatility of the short-rate itself, whereas the volatility parameter in the BDT Model is the volatility of the log short-rate.<br><br> The calibration process works via iterative solution of the theta parameter at each step, so that bond prices generated by the interest rate tree are consistent with the observed market term structure. Starting at the lower end, at each maturity the model computes the price of a zero coupon bond implied by the tree, compares it with the market price implied by the yield (or discount factor), and solves for the value of theta that aligns the two. Proceeding step by step across maturities ensures that the entire tree is calibrated so that model-implied bond values reproduce the given yield curve.
    </div>
    """, unsafe_allow_html=True)
