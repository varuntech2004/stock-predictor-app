import streamlit as st
import bcrypt
import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
from io import StringIO
import ta
# import database as db

# ---------------- Helper Functions ----------------
def fetch_stock_data(stock_symbol, data_period="1y"):
    return yf.download(stock_symbol, period=data_period)

def fetch_live_price(stock_symbol):
    stock = yf.Ticker(stock_symbol)
    live_price = stock.history(period="1d")["Close"].iloc[-1]
    return float(live_price)

def is_strong_password(password):
    return (len(password) >= 8 and
            re.search(r"[A-Z]", password) and
            re.search(r"[a-z]", password) and
            re.search(r"[0-9]", password) and
            re.search(r"[\W_]", password))

# ---------------- Enhanced Model with Technical Indicators ----------------
def enhanced_model(stock_data):
    df = stock_data.copy()
    
    # Existing features
    for col in ['Open', 'High', 'Low', 'Close']:
        df[f'{col}_SMA_5'] = df[col].rolling(5).mean()
        df[f'{col}_Lag_1'] = df[col].shift(1)
        df[f'{col}_Lag_2'] = df[col].shift(2)

    # Added Technical Indicators via `ta`
    # Relative Strength Index (RSI)
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    # Moving Average Convergence Divergence (MACD)
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()

    df.dropna(inplace=True)

    X = df[[
        'Open_SMA_5', 'High_SMA_5', 'Low_SMA_5', 'Close_SMA_5',
        'Open_Lag_1', 'Open_Lag_2',
        'High_Lag_1', 'High_Lag_2',
        'Low_Lag_1', 'Low_Lag_2',
        'Close_Lag_1', 'Close_Lag_2',
        'Volume',
        'RSI', 'MACD', 'MACD_Signal', 'BB_High', 'BB_Low'
    ]]
    y_cols = ['Open', 'High', 'Low', 'Close']
    y = df[y_cols]

    if X.empty:
        return pd.DataFrame(), 0, 0, 0, 0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    if len(X_test) > 0:
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred, multioutput='uniform_average')
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
    else:
        r2, mae, mse, rmse = 0, 0, 0, 0

    # Forecast next 7 days
    future_predictions = []
    # Work on a copy of the sequence to calculate future rolling windows correctly
    historical_last_30 = df.tail(30).copy() 
    
    for i in range(7):
        # We need at least the past few days to calculate SMA and Lags
        # For simplified forecasting of indicators like MACD/RSI, we use the last available rows and append predictions.
        # This is an approximation as calculating exact ta features for step+n requires the full series.
        
        features = {
            'Open_SMA_5': historical_last_30['Open'].tail(5).mean(),
            'High_SMA_5': historical_last_30['High'].tail(5).mean(),
            'Low_SMA_5': historical_last_30['Low'].tail(5).mean(),
            'Close_SMA_5': historical_last_30['Close'].tail(5).mean(),
            'Open_Lag_1': historical_last_30['Open'].iloc[-1],
            'Open_Lag_2': historical_last_30['Open'].iloc[-2],
            'High_Lag_1': historical_last_30['High'].iloc[-1],
            'High_Lag_2': historical_last_30['High'].iloc[-2],
            'Low_Lag_1': historical_last_30['Low'].iloc[-1],
            'Low_Lag_2': historical_last_30['Low'].iloc[-2],
            'Close_Lag_1': historical_last_30['Close'].iloc[-1],
            'Close_Lag_2': historical_last_30['Close'].iloc[-2],
            'Volume': historical_last_30['Volume'].iloc[-1],
            # Approximate current indicators to pass into feature predicting future
            'RSI': historical_last_30['RSI'].iloc[-1],
            'MACD': historical_last_30['MACD'].iloc[-1],
            'MACD_Signal': historical_last_30['MACD_Signal'].iloc[-1],
            'BB_High': historical_last_30['BB_High'].iloc[-1],
            'BB_Low': historical_last_30['BB_Low'].iloc[-1]
        }
        
        X_future = pd.DataFrame([features])
        y_future = model.predict(X_future)[0]
        prediction = dict(zip(y_cols, y_future))
        future_predictions.append(prediction)

        # Append new prediction to the end of our working dataframe to shift lags for the next step
        new_row_dict = {
            'Open': prediction['Open'],
            'High': prediction['High'],
            'Low': prediction['Low'],
            'Close': prediction['Close'],
            'Volume': features['Volume']
        }
        new_row_df = pd.DataFrame([new_row_dict])
        
        historical_last_30 = pd.concat([historical_last_30, new_row_df], ignore_index=True)
        # Recalculate indicators at the tail. This is computationally OK for small tails but RSI/MACD prefer long histories.
        # For a robust approach, we append to the whole DF and recalculate TAs
        full_temp_close = historical_last_30['Close']
        historical_last_30['RSI'] = ta.momentum.RSIIndicator(full_temp_close, window=14).rsi().fillna(features['RSI'])
        macd_obj = ta.trend.MACD(full_temp_close)
        historical_last_30['MACD'] = macd_obj.macd().fillna(features['MACD'])
        historical_last_30['MACD_Signal'] = macd_obj.macd_signal().fillna(features['MACD_Signal'])
        bb_obj = ta.volatility.BollingerBands(full_temp_close, window=20, window_dev=2)
        historical_last_30['BB_High'] = bb_obj.bollinger_hband().fillna(features['BB_High'])
        historical_last_30['BB_Low'] = bb_obj.bollinger_lband().fillna(features['BB_Low'])
        
    future_dates = [date.today() + timedelta(days=i) for i in range(1, 8)]
    pred_df = pd.DataFrame(future_predictions)
    pred_df["Date"] = future_dates

    return pred_df, r2, mae, mse, rmse

# ---------------- Authentication via PostgreSQL ----------------
def register_user():
    st.subheader("🔏 Register New User")
    new_name = st.text_input("Full Name")
    new_username = st.text_input("Username")
    new_password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if new_password and not is_strong_password(new_password):
        st.warning("Weak password. Use uppercase, lowercase, number, and symbol. (Min 8 chars)")

    if st.button("Register"):
        if not all([new_name, new_username, new_password, confirm_password]):
            st.warning("⚠️ Fill all fields.")
            return
        if new_password != confirm_password:
            st.error("❌ Passwords do not match.")
            return
        if not is_strong_password(new_password):
            st.error("❌ Weak password format.")
            return

        st.success("Registered successfully (demo mode)")
return
        if success:
            st.success("✅ Registered! Now login.")
        else:
            st.error("❌ Username already exists or database error.")

def login_user():
    st.subheader("🔑 Login to your account")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_pass")

    if st.button("Login"):
        if not username or not password:
            st.warning("⚠️ Fill in both fields.")
            return

        st.session_state["authenticated"] = True
st.session_state["name"] = username
st.success(f"Welcome {username}")
st.rerun()
        if user:
            st.session_state["authenticated"] = True
            st.session_state["user_id"] = user['id']
            st.session_state["username"] = user['username']
            st.session_state["name"] = user['name']
            st.success(f"✅ Welcome back, {st.session_state['name']}!")
            st.rerun()
        else:
            st.error("❌ Invalid Username or Password.")

def forgot_password():
    st.subheader("🔁 Forgot Password")
    username = st.text_input("Enter your Username")
    new_password = st.text_input("New Password", type="password")
    confirm_password = st.text_input("Confirm New Password", type="password")

    if st.button("Reset Password"):
        if not all([username, new_password, confirm_password]):
            st.warning("⚠️ Fill all fields.")
            return
        if new_password != confirm_password:
            st.error("❌ Passwords do not match.")
            return
        if not is_strong_password(new_password):
            st.error("❌ Weak password. Use uppercase, lowercase, number, and symbol.")
            return
            
        st.success("Password reset successful (demo)")
        if success:
            st.success("✅ Password reset successfully. Please login.")
        else:
            st.error("❌ Username not found or failed to reset.")

# ---------------- Main App ----------------
def main():
    st.set_page_config(page_title="📈 Pro Stock Predictor", layout="wide")

    # Initialize DB tables
   # db.init_db()

    if True:
        with st.sidebar:
            st.markdown(f"👋 Welcome, **{st.session_state['name']}**")
            st.title("📍 Select Stock")
            stock_exchange = st.selectbox("Stock Exchange", ["Indian Stock Market", "US Stock Market"])
            default_symbol = "RELIANCE.NS" if stock_exchange == "Indian Stock Market" else "AAPL"
            stock_symbol = st.text_input("Enter Stock Symbol", value=default_symbol)

            st.title("⏳ Chart Options")
            chart_data_period_options = ["1mo", "3mo", "6mo", "1y", "3y", "5y", "max"]
            chart_data_period = st.selectbox("Historical Data Range", chart_data_period_options, index=3)
            
            # --- Watchlist Section ---
            st.markdown("---")
            st.title("⭐️ Watchlist")
            user_id = st.session_state['user_id']
            if st.button("➕ Add Current to Watchlist", use_container_width=True):
                if stock_symbol:
                    if# db.add_to_watchlist(user_id, stock_symbol):
                        st.success(f"Added {stock_symbol}")
                    else:
                        st.error("Could not add or already exists.")
            
            watchlist =# db.get_watchlist(user_id)
            if watchlist:
                for item in watchlist:
                    col1, col2 = st.columns([0.7, 0.3])
                    with col1:
                        if st.button(f"🔎 {item}", key=f"load_{item}", use_container_width=True):
                            stock_symbol = item
                    with col2:
                        if st.button("❌", key=f"del_{item}"):
                           # db.remove_from_watchlist(user_id, item)
                            st.rerun()
            else:
                st.info("Watchlist is empty.")

            st.markdown("---")
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.clear()
                st.rerun()

        stock_data = pd.DataFrame()
        stock_data_for_charts = pd.DataFrame()
        live_price = None
        currency_symbol = "₹" if stock_exchange == "Indian Stock Market" else "$"

        if stock_symbol:
            with st.spinner(f"Loading data for {stock_symbol}..."):
                stock_data = fetch_stock_data(stock_symbol, "1y")
                stock_data_for_charts = fetch_stock_data(stock_symbol, chart_data_period)
                if isinstance(stock_data.columns, pd.MultiIndex):
                    stock_data.columns = stock_data.columns.droplevel(1)
                if isinstance(stock_data_for_charts.columns, pd.MultiIndex):
                    stock_data_for_charts.columns = stock_data_for_charts.columns.droplevel(1)
                
                try:
                    live_price = fetch_live_price(stock_symbol)
                except Exception as e:
                    live_price = None
                    
        # Proceed if we have basic data
        if stock_symbol and not stock_data.empty:
            
            # Predictive Analysis first so we can display recommendations in the top bar
            prediction_df, r2, mae, mse, rmse = enhanced_model(stock_data.copy())
            
            # Setup Top Metrics Dashboard
            st.title(f"Dashboard: {stock_symbol}")
            
            pred_next_day = None
            if not prediction_df.empty:
                pred_next_day = prediction_df.iloc[0]['Close']

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(label="Live Price", value=f"{currency_symbol}{live_price:.2f}" if live_price else "N/A")
            
            with col2:
                if pred_next_day:
                    price_diff = pred_next_day - (live_price if live_price else pred_next_day)
                    st.metric(label="Tomorrow Predicted Close", 
                              value=f"{currency_symbol}{pred_next_day:.2f}",
                              delta=f"{price_diff:.2f}")
                else:
                    st.metric(label="Tomorrow Predicted Close", value="N/A")
            
            with col3:
                # Recommendation Logic
                if live_price and pred_next_day:
                    # Let's consider > 0.5% a strong signal, otherwise Hold
                    threshold = live_price * 0.005
                    diff = pred_next_day - live_price
                    if diff > threshold:
                        st.metric(label="Action Recommendation", value="BUY 🟢")
                    elif diff < -threshold:
                        st.metric(label="Action Recommendation", value="SELL 🔴")
                    else:
                        st.metric(label="Action Recommendation", value="HOLD 🟡")
                else:
                     st.metric(label="Action Recommendation", value="N/A")
                     
            with col4:
                st.metric(label="Model R² Accuracy", value=f"{r2 * 100:.1f}%")

            st.markdown("---")

            # --- Interactive Candlestick Chart with SMAs ---
            st.subheader("📊 Technical Chart (Price & Moving Averages)")
            if not stock_data_for_charts.empty:
                fig = go.Figure()
                
                # Candlesticks
                fig.add_trace(go.Candlestick(x=stock_data_for_charts.index,
                                open=stock_data_for_charts['Open'],
                                high=stock_data_for_charts['High'],
                                low=stock_data_for_charts['Low'],
                                close=stock_data_for_charts['Close'],
                                name='Price'))
                                
                # Add SMA 20 and SMA 50 Overlays
                stock_data_for_charts['SMA_20'] = stock_data_for_charts['Close'].rolling(window=20).mean()
                stock_data_for_charts['SMA_50'] = stock_data_for_charts['Close'].rolling(window=50).mean()
                
                fig.add_trace(go.Scatter(x=stock_data_for_charts.index, y=stock_data_for_charts['SMA_20'],
                                         line=dict(color='blue', width=1.5), name='SMA 20'))
                fig.add_trace(go.Scatter(x=stock_data_for_charts.index, y=stock_data_for_charts['SMA_50'],
                                         line=dict(color='orange', width=1.5), name='SMA 50'))

                fig.update_layout(title=f"{stock_symbol} - {chart_data_period}", height=600, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # Layout for Data and Predictions
            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.subheader(f"📅 Historical Data (Last 5 Days)")
                st.dataframe(stock_data_for_charts.tail(), use_container_width=True)
                
                st.subheader("📉 Volume Profile")
                st.area_chart(stock_data_for_charts["Volume"].tail(60))

            with col_b:
                st.subheader("🔮 7-Day Stock Price Forecast")
                if not prediction_df.empty:
                    prediction_df_disp = prediction_df.set_index("Date")
                    st.line_chart(prediction_df_disp[["Open", "High", "Low", "Close"]])
                    st.dataframe(prediction_df_disp, use_container_width=True)
                
                st.write("**Model Evaluation Metrics**")
                st.write(f"- **MAE:** {mae:.2f}")
                st.write(f"- **MSE:** {mse:.2f}")
                st.write(f"- **RMSE:** {rmse:.2f}")

                # Download Report
                def generate_text_report(stock_symbol, live_price, historical_data, prediction_data, r2, mae, mse, rmse, currency_symbol):
                    report = f"Stock Report for {stock_symbol}\n\n"
                    if live_price:
                        report += f"Live Price: {currency_symbol}{live_price:.2f}\n\n"
                    report += "Historical Data (Last 5 Days):\n"
                    report += historical_data.tail().to_string() + "\n\n"
                    report += "Prediction Data (Next 7 Days):\n"
                    report += prediction_data.to_string() + "\n\n"
                    report += "Model Evaluation:\n"
                    report += f"  R² Score (Accuracy %): {r2 * 100:.2f}%\n"
                    report += f"  MAE: {mae:.2f}\n"
                    report += f"  RMSE: {rmse:.2f}\n"
                    return report

                if not prediction_df.empty:
                    text_report = generate_text_report(stock_symbol, live_price, stock_data, prediction_df, r2, mae, mse, rmse, currency_symbol)
                    st.download_button(
                        label="📄 Download Text Report",
                        data=text_report,
                        file_name=f"{stock_symbol}_report_{date.today().strftime('%Y-%m-%d')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

        elif stock_symbol:
            st.warning("No data found for this ticker. Please check the symbol.")

    else:
        st.title("📈 Pro Stock Predictor")
        st.markdown("Analyze markets, use machine learning models, and build your watchlist securely.")
        
        st.title("📈 Pro Stock Predictor 🚀")
st.success("App running without login system")
        with tab1:
            login_user()
        with tab2:
            register_user()
        with tab3:
            forgot_password()

if __name__ == "__main__":
    main()