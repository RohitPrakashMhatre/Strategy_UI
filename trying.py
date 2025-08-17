import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import vectorbt as vbt
import datetime as dt
import numpy as np
import plotly.graph_objects as go
# import streamlit_authenticator as stauth
# from auth_config import config

# # Authentication
# authenticator = stauth.Authenticate(
#     config["names"],
#     config["usernames"],
#     config["hashed_passwords"],
#     config["cookie_name"],
#     config["signature_key"],
#     cookie_expiry_days=config["cookie_expiry_days"]
# )

# name, authentication_status, username = authenticator.login("Login", "main")

# if authentication_status:
#     st.write(f"Welcome {name}!")
#     # Put your dashboard code here
# elif authentication_status == False:
#     st.error("Username/password is incorrect")
# elif authentication_status == None:
#     st.warning("Please enter your username and password")

st.set_page_config(page_title="Trading Dashboard", layout="wide")

st.title("📊 Trading Dashboard")

# --- Input Section ---
col1, col2 = st.columns(2)
with col1:
    symbol1 = st.text_input("Enter First Symbol", "^GSPC").upper()
with col2:
    symbol2 = st.text_input("Enter Second Symbol", "DX-Y.NYB").upper()
start_date = '2008-01-11'
end_date = dt.datetime.today()
# Function to fetch data
@st.cache_data(ttl=3600)
def get_data(symbol):
    return yf.download(symbol, start=start_date, end=end_date, interval="1d")

# --- Main Logic ---
if st.button("Fetch Data & Run Strategy"):
    data1 = get_data(symbol1)
    data2 = get_data(symbol2)

    if data1.empty or data2.empty:
        st.error("Could not fetch data for one or both symbols.")
    else:
        # constant paramerters
        window = 55
        stop_loss = None
        diff = 0.001
        ensemble_param = 2
        quantile = 0.7
        exp_span = 50

        df = pd.DataFrame({
            f"{symbol1}_Close": data1[("Close", symbol1)],
            f"{symbol2}_Close": data2[("Close", symbol2)]
        })
        # Calculate percent chagne
        df[f"{symbol1}_returns"] = df[f"{symbol1}_Close"].pct_change()
        df[f"{symbol2}_returns"] = df[f"{symbol2}_Close"].pct_change()
        df[f"{symbol1}_returns"] = df[f"{symbol1}_Close"].pct_change().fillna(0)
        df[f"{symbol2}_returns"] = df[f"{symbol2}_Close"].pct_change().fillna(0)
        # Simple net returns
        df[f"{symbol1}_net_returns"] = (df[f"{symbol1}_returns"].rolling(window=window).mean() /
                                        df[f"{symbol1}_returns"].rolling(window=window).std())
        df[f"{symbol2}_net_returns"] = (df[f"{symbol2}_returns"].rolling(window=window).mean() /
                                        df[f"{symbol2}_returns"].rolling(window=window).std()) 
        # exponential net returns
        df[f"{symbol1}_exp_net"] = (df[f"{symbol1}_returns"].ewm(span=exp_span).mean() /
                                   df[f"{symbol1}_returns"].ewm(span=exp_span).std())
        df[f"{symbol2}_exp_net"] = (df[f"{symbol2}_returns"].ewm(span=exp_span).mean() /
                                   df[f"{symbol2}_returns"].ewm(span=exp_span).std())
        df = df.dropna()

        # Momentum's
        df[f"{symbol1}_momentum"] = df[f"{symbol1}_returns"].rolling(window=window).mean()
        df[f"{symbol2}_momentum"] = df[f"{symbol2}_returns"].rolling(window=window).mean()

        # volatility
        df["volatility"] = df[f"{symbol1}_returns"].rolling(window=10).std()
        vol_threshold = df["volatility"].quantile(quantile)

        # Generating multiple signals
        df["old_net_signal"] = (df[f"{symbol1}_net_returns"] > df[f"{symbol2}_net_returns"]).astype(int)
        df["momentum_signal"] = (df[f"{symbol1}_momentum"] > df[f"{symbol2}_momentum"]).astype(int)
        df["short_signal_exp"] = (df[f"{symbol1}_exp_net"] > df[f"{symbol2}_exp_net"]).astype(int)
        df["abs_signal"] = np.where(
            abs(df[f"{symbol1}_exp_net"] - df[f"{symbol2}_exp_net"]) > diff, 1, 0
        )
        df["final_signal"] = np.where(
            (df["abs_signal"].shift(1) == 1) & (df["momentum_signal"].shift(1) == 1), 1, 1
        )
        df["vol_filter"] = (df["volatility"] < vol_threshold).astype(int)
        df["vol_signal"] = df["final_signal"] * df["vol_filter"]

        all_signals = [
            df["old_net_signal"],
            df["momentum_signal"],
            df["short_signal_exp"],
            df["abs_signal"],
            #df["final_signal"],
            df["vol_filter"],
            df["vol_signal"]
        ]
        df["ensemble_signal"] = (sum(all_signals) >= ensemble_param).astype(int)

        # Calculate simple returns & cumulative returns
        df["strategy_returns"] = df[f"{symbol1}_returns"] * df["ensemble_signal"].shift(1)
        df['strategy_returns'] = df['strategy_returns'].fillna(0)
        df["fstrategy_cumm_ret"] = ((1 + df["strategy_returns"]).cumprod() - 1) * 100
        df[f"{symbol1}_cumm_returns"] = ((1 + df[f"{symbol1}_returns"]).cumprod() - 1) * 100 

        entry = df["ensemble_signal"] == 1
        exits = df["ensemble_signal"] == 0

        entry = entry.reindex(df.index,method='bfill')
        exits = exits.reindex(df.index,method='bfill')

        portfolio = vbt.Portfolio.from_signals(
            close = df[f"{symbol1}_Close"],
            entries = entry,
            exits = exits,
            sl_stop = None,
            freq = "1D",
            init_cash = 10000
        )

        fig = go.Figure()

        # Plot closing price
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[f"{symbol1}_Close"],  # Change symbol1 to your column name
            mode='lines',
            name=f"{symbol1} Price"
        ))

        # Mark buy signals
        fig.add_trace(go.Scatter(
            x=df.index[df["ensemble_signal"] == 1],  # Where final_signal is 1
            y=df[f"{symbol1}_Close"][df["ensemble_signal"] == 1],
            mode='markers',
            marker=dict(color='green', size=10, symbol='triangle-up'),
            name='Buy Signal'
        ))

        # Mark sell signals
        fig.add_trace(go.Scatter(
            x=df.index[df["ensemble_signal"] == 0],  # Where final_signal is -1
            y=df[f"{symbol1}_Close"][df["ensemble_signal"] == 0],
            mode='markers',
            marker=dict(color='red', size=10, symbol='triangle-down'),
            name='Sell Signal'
        ))

        fig.update_layout(
            title=f"{symbol1} Price with Buy/Sell Signals {exp_span}",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Strategy Output Table")
        st.dataframe(df.tail(15))

        stats = portfolio.stats()
        st.subheader("Portfolio Statistics")
        st.dataframe(stats.to_frame(name="Value"))
else:
    st.info("Enter two symbols and click 'Fetch Data & Run Strategy'.")
