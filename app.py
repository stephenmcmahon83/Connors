import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="ConnorsRSI Pullback Scanner & Backtester",
    page_layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# TECHNICAL INDICATORS CALCULATION
# ==========================================
def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    """Standard Wilder's RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def compute_streak(series: pd.Series) -> pd.Series:
    """Calculates consecutive days of up/down closes."""
    streak = np.zeros(len(series))
    close_vals = series.values
    for i in range(1, len(close_vals)):
        if close_vals[i] > close_vals[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close_vals[i] < close_vals[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    return pd.Series(streak, index=series.index)

def compute_percent_rank(series: pd.Series, lookback: int = 100) -> pd.Series:
    """Percentage of prior 1-day returns in the lookback lower than today's return."""
    pct_change = series.pct_change()
    
    def rolling_rank(window):
        current_return = window[-1]
        prior_returns = window[:-1]
        return (prior_returns < current_return).mean() * 100.0
    
    return pct_change.rolling(lookback + 1).apply(rolling_rank, raw=True)

def compute_connors_rsi(df: pd.DataFrame, rsi_len: int = 3, streak_len: int = 2, rank_len: int = 100) -> pd.Series:
    """ConnorsRSI = (RSI(Close, 3) + RSI(Streak, 2) + PercentRank(Return, 100)) / 3."""
    price_rsi = compute_rsi(df['Close'], rsi_len)
    streak_series = compute_streak(df['Close'])
    streak_rsi = compute_rsi(streak_series, streak_len)
    prank = compute_percent_rank(df['Close'], rank_len)
    
    return (price_rsi + streak_rsi + prank) / 3.0

def compute_adx(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """10-day Average Directional Index (ADX) using Wilder smoothing."""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    tr_smooth = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_dm_smooth = pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    plus_di = 100 * (plus_dm_smooth / (tr_smooth + 1e-10))
    minus_di = 100 * (minus_dm_smooth / (tr_smooth + 1e-10))
    
    dx = 100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-10)
    adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return adx

def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Appends all required filters and indicator columns to the dataset."""
    df = df.copy()
    df['Vol_SMA21'] = df['Volume'].rolling(21).mean()
    df['ADX10'] = compute_adx(df, period=10)
    df['ConnorsRSI'] = compute_connors_rsi(df, 3, 2, 100)
    
    # Intraday low dip from yesterday's close
    df['Prev_Close'] = df['Close'].shift(1)
    df['Low_Drop_Pct'] = (1 - (df['Low'] / df['Prev_Close'])) * 100.0
    
    # Position of close relative to the day's total range (0 = at low, 100 = at high)
    day_range = df['High'] - df['Low']
    df['Close_Range_Pct'] = np.where(day_range > 0, ((df['Close'] - df['Low']) / day_range) * 100.0, 50.0)
    
    return df

# ==========================================
# SIDEBAR PARAMETERS
# ==========================================
st.sidebar.title("Strategy Parameters")

W = st.sidebar.selectbox("W: Intraday Low Drop Below Prior Close (%)", [2, 4, 6, 8], index=1)
X = st.sidebar.selectbox("X: Max Close Location in Day's Range (%)", [10, 25], index=1)
Y = st.sidebar.slider("Y: ConnorsRSI Entry Cutoff (< Y)", min_value=5, max_value=15, value=10, step=1)
Z = st.sidebar.selectbox("Z: Next-Day Intraday Limit Discount (%)", [4, 6, 8, 10], index=0)
N = st.sidebar.selectbox("N: ConnorsRSI Exit Threshold (> N)", [50, 60, 70, 80], index=2)

st.sidebar.markdown("---")
min_price = st.sidebar.number_input("Minimum Stock Price ($)", value=5.0, step=1.0)
min_volume = st.sidebar.number_input("Minimum 21-Day Avg Volume", value=250000, step=50000)

# ==========================================
# BACKTEST ENGINE
# ==========================================
def run_backtest(df: pd.DataFrame, W: float, X: float, Y: float, Z: float, N: float, min_price: float, min_volume: float):
    trades = []
    in_position = False
    pending_order = None
    position = {}

    # Iterate bar by bar
    for i in range(120, len(df)):
        row = df.iloc[i]
        date = df.index[i]

        # 1. Check if we hold a position -> Check Exit Rule
        if in_position:
            # Exit rule: Close position when ConnorsRSI > N at closing price
            if row['ConnorsRSI'] > N:
                exit_price = row['Close']
                pnl = (exit_price - position['entry_price']) / position['entry_price']
                trades.append({
                    'Entry Date': position['entry_date'].strftime('%Y-%m-%d'),
                    'Entry Price': position['entry_price'],
                    'Exit Date': date.strftime('%Y-%m-%d'),
                    'Exit Price': exit_price,
                    'Return (%)': round(pnl * 100, 2),
                    'Holding Days': (date - position['entry_date']).days
                })
                in_position = False
                position = {}
            continue

        # 2. Check if a limit order from yesterday gets filled today
        if pending_order is not None:
            limit_price = pending_order['limit_price']
            # Order triggers if today's low hits or breaches the limit price
            if row['Low'] <= limit_price:
                # If market opens below the limit, execution occurs at the better open price
                fill_price = row['Open'] if row['Open'] < limit_price else limit_price
                in_position = True
                position = {
                    'entry_date': date,
                    'entry_price': fill_price
                }
                pending_order = None
                
                # Check if exit condition triggers on entry day close
                if row['ConnorsRSI'] > N:
                    pnl = (row['Close'] - fill_price) / fill_price
                    trades.append({
                        'Entry Date': date.strftime('%Y-%m-%d'),
                        'Entry Price': fill_price,
                        'Exit Date': date.strftime('%Y-%m-%d'),
                        'Exit Price': row['Close'],
                        'Return (%)': round(pnl * 100, 2),
                        'Holding Days': 0
                    })
                    in_position = False
                    position = {}
                continue
            else:
                # 1-day limit order expires unfilled
                pending_order = None

        # 3. Check Setup Rules on Today's Bar
        cond1 = row['Close'] > min_price
        cond2 = row['Vol_SMA21'] >= min_volume
        cond3 = row['ADX10'] > 30.0
        cond4 = row['Low_Drop_Pct'] >= W
        cond5 = row['Close_Range_Pct'] <= X
        cond6 = row['ConnorsRSI'] < Y

        if cond1 and cond2 and cond3 and cond4 and cond5 and cond6:
            pending_order = {
                'limit_price': round(row['Close'] * (1 - Z / 100.0), 2),
                'signal_date': date
            }

    return pd.DataFrame(trades)

# ==========================================
# APP TABS: SCANNER & BACKTESTER
# ==========================================
tab_scanner, tab_backtest, tab_rules = st.tabs(["Candidate Scanner (For Tomorrow)", "Strategy Backtester", "Strategy Rules"])

# ------------------------------------------
# TAB 1: SCANNER
# ------------------------------------------
with tab_scanner:
    st.subheader("Daily Pullback Candidate Scanner")
    st.write("Scans for stocks satisfying all setup criteria on today's close and computes tomorrow's intraday limit buy price.")
    
    default_tickers = "AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, AMD, QQQ, SPY, IWM, NFLX, COIN, PLTR, INTC"
    ticker_input = st.text_area("Enter Ticker Symbols (comma-separated):", value=default_tickers)
    scan_button = st.button("Run Scanner", type="primary")

    if scan_button:
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        candidates = []
        scan_progress = st.progress(0)

        for idx, ticker in enumerate(tickers):
            try:
                stock_df = yf.download(ticker, period="1y", interval="1d", progress=False)
                if isinstance(stock_df.columns, pd.MultiIndex):
                    stock_df.columns = stock_df.columns.get_level_values(0)

                if len(stock_df) >= 120:
                    stock_df = prepare_indicators(stock_df)
                    latest = stock_df.iloc[-1]
                    
                    # Strategy Conditions
                    c1 = latest['Close'] > min_price
                    c2 = latest['Vol_SMA21'] >= min_volume
                    c3 = latest['ADX10'] > 30.0
                    c4 = latest['Low_Drop_Pct'] >= W
                    c5 = latest['Close_Range_Pct'] <= X
                    c6 = latest['ConnorsRSI'] < Y

                    if c1 and c2 and c3 and c4 and c5 and c6:
                        limit_target = round(latest['Close'] * (1 - Z / 100.0), 2)
                        candidates.append({
                            "Ticker": ticker,
                            "Today Close ($)": round(latest['Close'], 2),
                            f"Tomorrow Limit ({Z}% off)": limit_target,
                            "ConnorsRSI": round(latest['ConnorsRSI'], 2),
                            "ADX(10)": round(latest['ADX10'], 2),
                            "Low Drop vs Prev Close (%)": round(latest['Low_Drop_Pct'], 2),
                            "Close Range Loc (%)": round(latest['Close_Range_Pct'], 2),
                            "21D Avg Vol": f"{int(latest['Vol_SMA21']):,}"
                        })
            except Exception as e:
                st.warning(f"Error fetching data for {ticker}: {e}")

            scan_progress.progress((idx + 1) / len(tickers))

        if candidates:
            res_df = pd.DataFrame(candidates)
            st.success(f"Found {len(candidates)} trigger setups meeting all criteria!")
            st.dataframe(res_df, use_container_width=True)
        else:
            st.info("No candidates triggered the strategy on the most recent completed trading session.")

# ------------------------------------------
# TAB 2: BACKTESTER
# ------------------------------------------
with tab_backtest:
    st.subheader("Historical Backtester")
    col_sym, col_start, col_end = st.columns([1, 1, 1])
    with col_sym:
        bt_ticker = st.text_input("Ticker Symbol", value="QQQ").upper().strip()
    with col_start:
        start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=5 * 365))
    with col_end:
        end_date = st.date_input("End Date", value=datetime.today())

    bt_button = st.button("Run Backtest", type="primary")

    if bt_button:
        with st.spinner(f"Loading data and backtesting {bt_ticker}..."):
            raw_data = yf.download(bt_ticker, start=start_date, end=end_date, progress=False)
            if isinstance(raw_data.columns, pd.MultiIndex):
                raw_data.columns = raw_data.columns.get_level_values(0)

            if len(raw_data) < 150:
                st.error("Insufficient historical data for the selected date range.")
            else:
                data_with_indicators = prepare_indicators(raw_data)
                trade_log = run_backtest(
                    data_with_indicators,
                    W=W, X=X, Y=Y, Z=Z, N=N,
                    min_price=min_price,
                    min_volume=min_volume
                )

                if trade_log.empty:
                    st.warning("No trades were triggered with the current parameters.")
                else:
                    # Performance Metrics
                    total_trades = len(trade_log)
                    wins = trade_log[trade_log['Return (%)'] > 0]
                    win_rate = (len(wins) / total_trades) * 100.0
                    avg_return = trade_log['Return (%)'].mean()
                    total_comp_return = (np.prod(1 + trade_log['Return (%)'] / 100.0) - 1) * 100.0
                    
                    gross_profit = trade_log[trade_log['Return (%)'] > 0]['Return (%)'].sum()
                    gross_loss = abs(trade_log[trade_log['Return (%)'] < 0]['Return (%)'].sum())
                    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else np.nan

                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Total Trades", total_trades)
                    m2.metric("Win Rate", f"{win_rate:.1f}%")
                    m3.metric("Profit Factor", profit_factor if not np.isnan(profit_factor) else "∞")
                    m4.metric("Avg Trade Return", f"{avg_return:.2f}%")
                    m5.metric("Compound Strategy Return", f"{total_comp_return:.1f}%")

                    # Cumulative Equity Curve
                    trade_log['Cumulative Equity'] = (1 + trade_log['Return (%)'] / 100.0).cumprod() * 100.0
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=trade_log['Exit Date'],
                        y=trade_log['Cumulative Equity'],
                        mode='lines+markers',
                        name='Equity Curve',
                        line=dict(color='#2962FF', width=2)
                    ))
                    fig.update_layout(
                        title=f"{bt_ticker} - Strategy Equity Curve (Base = 100)",
                        xaxis_title="Exit Date",
                        yaxis_title="Normalized Portfolio Value",
                        template="plotly_white",
                        height=450
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.write("### Trade Log")
                    st.dataframe(trade_log, use_container_width=True)

# ------------------------------------------
# TAB 3: STRATEGY RULES
# ------------------------------------------
with tab_rules:
    st.markdown("""
    ### ConnorsRSI Pullback Strategy Architecture
    
    1. **Price Floor:** Stock price must be above **$5.00** per share.
    2. **Liquidity Filter:** 21-day simple moving average volume $\ge$ **250,000 shares/day**.
    3. **Trend Strength:** 10-day Average Directional Index (ADX) > **30.0**.
    4. **Intraday Washout:** Today's Low drops at least **W%** ($W \in \{2, 4, 6, 8\}$) below yesterday's closing price.
    5. **Weak Close:** Today's Close finishes in the bottom **X%** ($X \in \{10, 25\}$) of today's total range $(High - Low)$.
    6. **Deep Oversold Oscillator:** ConnorsRSI(3, 2, 100) < **Y** ($Y \in [5, 15]$).
    7. **Execution (Tomorrow):** Place an intraday limit order **Z%** ($Z \in \{4, 6, 8, 10\}$) below today's close.
    8. **Exit:** Liquidate the position at the close on the first day ConnorsRSI(3, 2, 100) closes above **N** ($N \in \{50, 60, 70, 80\}$).
    """)
