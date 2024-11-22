import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(
    page_title="American Option Pricing Binary TreeModel",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded")


st.markdown("""
<style>
/* Adjust the size and alignment of the CALL and PUT value containers */
.metric-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 8px; /* Adjust the padding to control height */
    width: auto; /* Auto width for responsiveness, or set a fixed width if necessary */
    margin: 0 auto; /* Center the container */
}

/* Custom classes for CALL and PUT values */
.metric-call {
    background-color: #90ee90; /* Light green background */
    color: black; /* Black font color */
    margin-right: 10px; /* Spacing between CALL and PUT */
    border-radius: 10px; /* Rounded corners */
}

.metric-put {
    background-color: #ffcccb; /* Light red background */
    color: black; /* Black font color */
    border-radius: 10px; /* Rounded corners */
}

/* Style for the value text */
.metric-value {
    font-size: 1.5rem; /* Adjust font size */
    font-weight: bold;
    margin: 0; /* Remove default margins */
}

/* Style for the label text */
.metric-label {
    font-size: 1rem; /* Adjust font size */
    margin-bottom: 4px; /* Spacing between label and value */
}

</style>
""", unsafe_allow_html=True)

# (Include the BlackScholes class definition here)
class AmericanOption:
    def __init__(
        self,
        time_to_maturity: float,
        strike: float,
        current_price: float,
        volatility: float,
        interest_rate: float,
        steps: int = 100
    ):
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate
        self.steps = steps

    def calculate_prices(self):
        T = self.time_to_maturity
        S = self.current_price
        K = self.strike
        r = self.interest_rate
        σ = self.volatility
        N = self.steps

        # Time increment per step
        dt = T / N
        u = np.exp(σ * np.sqrt(dt))  # Up factor
        d = 1 / u  # Down factor
        p = (np.exp(r * dt) - d) / (u - d)  # Risk-neutral probability

        # Initialize asset prices at maturity
        prices = np.zeros(N + 1)
        for i in range(N + 1):
            prices[i] = S * (u**i) * (d**(N - i))

        # Initialize option values at maturity
        call_values = np.maximum(prices - K, 0)  # Call payoff
        put_values = np.maximum(K - prices, 0)  # Put payoff

        # Step back through the tree
        for step in range(N - 1, -1, -1):
            for i in range(step + 1):
                call_values[i] = max(
                    prices[i] - K,  # Early exercise for call
                    np.exp(-r * dt) * (p * call_values[i + 1] + (1 - p) * call_values[i])
                )
                put_values[i] = max(
                    K - prices[i],  # Early exercise for put
                    np.exp(-r * dt) * (p * put_values[i + 1] + (1 - p) * put_values[i])
                )
                prices[i] = prices[i] * d

        self.call_price = call_values[0]
        self.put_price = put_values[0]

        return self.call_price, self.put_price
with st.sidebar:
    st.title("📊 American Options Pricing")
    st.write("Customize your parameters below:")

    # User inputs
    current_price = st.number_input("Current Price of Underlying Asset (S)", value=100.0, step=1.0)
    strike = st.number_input("Strike Price (K)", value=90.0, step=1.0)
    time_to_maturity = st.number_input("Time to Maturity (in years, T)", value=1.0, step=0.1)
    volatility = st.number_input("Volatility (σ)", value=0.2, step=0.01)
    interest_rate = st.number_input("Risk-Free Interest Rate (r)", value=0.05, step=0.01)
    steps = st.slider("Number of Steps in Binomial Tree", min_value=10, max_value=500, value=100, step=10)
# Create an instance of the AmericanOption class
model = AmericanOption(
    time_to_maturity=time_to_maturity,
    strike=strike,
    current_price=current_price,
    volatility=volatility,
    interest_rate=interest_rate,
    steps=steps
)

# Calculate call and put prices
call_price, put_price = model.calculate_prices()

# Display the results
st.subheader("Option Prices")
st.metric(label="Call Price", value=f"${call_price:.2f}")
st.metric(label="Put Price", value=f"${put_price:.2f}")

def plot_heatmap(model, spot_range, vol_range):
    call_prices = np.zeros((len(vol_range), len(spot_range)))
    put_prices = np.zeros((len(vol_range), len(spot_range)))

    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            temp_model = AmericanOption(
                time_to_maturity=model.time_to_maturity,
                strike=model.strike,
                current_price=spot,
                volatility=vol,
                interest_rate=model.interest_rate,
                steps=model.steps
            )
            temp_model.calculate_prices()
            call_prices[i, j] = temp_model.call_price
            put_prices[i, j] = temp_model.put_price

    # Call Price Heatmap
    fig_call, ax_call = plt.subplots(figsize=(10, 8))
    sns.heatmap(call_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f", cmap="viridis", ax=ax_call)
    ax_call.set_title("Call Price Heatmap")
    ax_call.set_xlabel("Spot Price")
    ax_call.set_ylabel("Volatility")

    # Put Price Heatmap
    fig_put, ax_put = plt.subplots(figsize=(10, 8))
    sns.heatmap(put_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f", cmap="viridis", ax=ax_put)
    ax_put.set_title("Put Price Heatmap")
    ax_put.set_xlabel("Spot Price")
    ax_put.set_ylabel("Volatility")

    return fig_call, fig_put

# Generate heatmaps
spot_min, spot_max = current_price * 0.8, current_price * 1.2
vol_min, vol_max = volatility * 0.5, volatility * 1.5
spot_range = np.linspace(spot_min, spot_max, 10)
vol_range = np.linspace(vol_min, vol_max, 10)

fig_call, fig_put = plot_heatmap(model, spot_range, vol_range)

# Display heatmaps
col1, col2 = st.columns(2)
with col1:
    st.pyplot(fig_call)
with col2:
    st.pyplot(fig_put)
