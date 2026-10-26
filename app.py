import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(
    page_title="ConnorsRSI 2-Day Bucket Backtester",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 1. INDICATORS & CRSI CALCULATIONS
# ==========================================

def calculate_rsi(series: pd.Series, period: int) -> pd.Series:
    """Wilder's RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_streak(close: pd.Series) -> pd.Series:
    """Calculates consecutive days of higher closes (+) or lower closes (-)."""
    diff = close.diff()
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if diff.iloc[i] > 0:
            streak[i] = streak[i - 1] + 1 if streak[i - 1] > 0 else 1
        elif diff.iloc[i] < 0:
            streak[i] = streak[i - 1] - 1 if streak[i - 1] < 0 else -1
        else:
            streak[i] = 0
    return pd.Series(streak, index=close.index)

def calculate_percent_rank(series: pd.Series, period: int = 100) -> pd.Series:
    """Rolling 1-day return percentile rank over N periods (0 to 100)."""
    pct_change = series.pct_change()
    
    def rolling_rank(window):
        current = window[-1]
        return (np.sum(window[:-1] < current) / (len(window) - 1)) * 100 if len(window) > 1 else 50.0

    return pct_change.rolling(period + 1).apply(rolling_rank, raw=True).fillna(50)

def prepare_crsi_data(df: pd.DataFrame, x_days: int) -> pd.DataFrame:
    """Computes ConnorsRSI(3,2,100), its 2-day SMA, and rolling X-day highs."""
    df = df.copy()
    rsi_p = calculate_rsi(df['Close'], 3)
    rsi_s = calculate_rsi(calculate_streak(df['Close']), 2)
    pr = calculate_percent_rank(df['Close'], 100)
    
    df['CRSI'] = (rsi_p + rsi_s + pr) / 3.0
    df['CRSI_2D'] = df['CRSI'].rolling(2).mean()
    
    # Prior X-day highest high (excluding current bar to avoid lookahead)
    df['X_Day_High'] = df['High'].shift(1).rolling(x_days).max()
    return df

# ==========================================
# 2. BUCKET DEFINITIONS & BACKTEST ENGINE
# ==========================================

BUCKET_RANGES = {
    "0 - 10": (0.0, 10.0),
    "11 - 20": (10.0001, 20.0),
    "21 - 30": (20.0001, 30.0),
    "31 - 40": (30.0001, 40.0),
    "41 - 50": (40.0001, 50.0),
    "51 - 60": (50.0001, 60.0),
    "61 - 70": (60.0001, 70.0),
    "71 - 80": (70.0001, 80.0),
    "81 - 90": (80.0001, 90.0),
    "91 - 100": (90.0001, 100.0),
}

def backtest_bucket(
    df: pd.DataFrame, 
    low_b: float, 
    high_b: float, 
    exit_mode: str, 
    crsi_exit_thresh: float, 
    entry_timing: str = "Next Open",
    initial_capital: float = 10000.0
):
    trades = []
    in_trade = False
    entry_price = 0.0
    entry_date = None
    shares = 0
    capital = initial_capital
    portfolio = []
    
    dates = df.index
    n = len(df)
    
    # Signal mask: 2-day CRSI inside bucket
    signal = (df['CRSI_2D'] >= low_b) & (df['CRSI_2D'] <= high_b)
    
    for i in range(1, n):
        curr_date = dates[i]
        curr_close = df['Close'].iloc[i]
        curr_open = df['Open'].iloc[i]
        curr_high = df['High'].iloc[i]
        curr_crsi_2d = df['CRSI_2D'].iloc[i]
        x_high = df['X_Day_High'].iloc[i]
        
        # 1. Manage Active Trade Exits
        if in_trade:
            hit_crsi = curr_crsi_2d > crsi_exit_thresh
            hit_x_high = pd.notna(x_high) and (curr_high >= x_high)
            
            should_exit = False
            exit_price = curr_close
            exit_reason = ""
            
            if exit_mode == "Either (Whichever occurs first)":
                if hit_x_high and hit_crsi:
                    exit_price = max(curr_open, x_high)
                    exit_reason = "Both (X-High & CRSI)"
                    should_exit = True
                elif hit_x_high:
                    exit_price = max(curr_open, x_high)
                    exit_reason = "X-Day High"
                    should_exit = True
                elif hit_crsi:
                    exit_price = curr_close
                    exit_reason = f"CRSI 2D > {crsi_exit_thresh}"
                    should_exit = True
            elif exit_mode == "Both Required":
                if hit_x_high and hit_crsi:
                    exit_price = curr_close
                    exit_reason = "Both Met"
                    should_exit = True
            elif exit_mode == "X-Day High Only":
                if hit_x_high:
                    exit_price = max(curr_open, x_high)
                    exit_reason = "X-Day High"
                    should_exit = True
            elif exit_mode == "2-Day CRSI Threshold Only":
                if hit_crsi:
                    exit_price = curr_close
                    exit_reason = f"CRSI 2D > {crsi_exit_thresh}"
                    should_exit = True
                    
            if should_exit:
                pnl = (exit_price - entry_price) * shares
                ret_pct = ((exit_price / entry_price) - 1) * 100
                capital += (shares * exit_price)
                
                trades.append({
                    'Entry Date': entry_date,
                    'Entry Price': round(entry_price, 2),
                    'Exit Date': curr_date,
                    'Exit Price': round(exit_price, 2),
                    'Exit Reason': exit_reason,
                    'Hold Days': (curr_date - entry_date).days,
                    'Return %': round(ret_pct, 2),
                    'PnL ($)': round(pnl, 2),
                    'Equity': round(capital, 2)
                })
                in_trade = False
                shares = 0

        # 2. Check for New Entries
        if not in_trade:
            if entry_timing == "Next Open" and signal.iloc[i - 1]:
                entry_price = curr_open
                entry_date = curr_date
                shares = int(capital // entry_price)
                if shares > 0:
                    capital -= (shares * entry_price)
                    in_trade = True
            elif entry_timing == "Signal Day Close" and signal.iloc[i]:
                entry_price = curr_close
                entry_date = curr_date
                shares = int(capital // entry_price)
                if shares > 0:
                    capital -= (shares * entry_price)
                    in_trade = True
                    
        curr_eq = capital + (shares * curr_close if in_trade else 0)
        portfolio.append({'Date': curr_date, 'Equity': curr_eq})
        
    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(portfolio).set_index('Date') if portfolio else pd.DataFrame()
    return trades_df, equity_df

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================

st.sidebar.title("Configuration")

ticker = st.sidebar.text_input("Ticker Symbol", value="SPY").upper()
start_date = st.sidebar.date_input("Start Date", value=datetime.today() - timedelta(days=10*365))
capital = st.sidebar.number_input("Starting Capital ($)", value=10000.0, step=1000.0)

st.sidebar.markdown("---")
st.sidebar.subheader("Exit Rules")

exit_mode = st.sidebar.selectbox(
    "Exit Condition Logic",
    [
        "Either (Whichever occurs first)",
        "X-Day High Only",
        "2-Day CRSI Threshold Only",
        "Both Required"
    ]
)

x_days = st.sidebar.slider("X-Day High Period (Days)", min_value=1, max_value=20, value=5, step=1)

crsi_exit_thresh = st.sidebar.selectbox(
    "2-Day CRSI Exit Threshold",
    options=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
    index=7  # Default 70
)

entry_timing = st.sidebar.radio("Entry Execution", ["Next Open", "Signal Day Close"], horizontal=True)

# ==========================================
# 4. APP WORKFLOW
# ==========================================

@st.cache_data(ttl=3600)
def load_data(symbol: str, start: datetime):
    df = yf.download(symbol, start=start, interval="1d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

raw_df = load_data(ticker, start_date)

if raw_df.empty or len(raw_df) < 120:
    st.error("Insufficient market data. Please verify the ticker or pick an earlier start date.")
    st.stop()

processed_df = prepare_crsi_data(raw_df, x_days=x_days)

tab1, tab2 = st.tabs(["📊 All Buckets Study (Larry Connors Matrix)", "📈 Single Bucket Deep Dive"])

# ------------------------------------------
# TAB 1: ALL BUCKETS MATRIX
# ------------------------------------------
with tab1:
    st.subheader(f"2-Day ConnorsRSI Bucket Study — {ticker}")
    st.caption(
        f"Comparing trade performance when entering on specific 2-Day CRSI bands. "
        f"Exit Mode: {exit_mode} | X-Day High: {x_days} | CRSI Exit Threshold: > {crsi_exit_thresh}"
    )
    
    matrix_rows = []
    bucket_results = {}
    
    for b_name, (lb, hb) in BUCKET_RANGES.items():
        tdf, eqdf = backtest_bucket(
            df=processed_df,
            low_b=lb,
            high_b=hb,
            exit_mode=exit_mode,
            crsi_exit_thresh=crsi_exit_thresh,
            entry_timing=entry_timing,
            initial_capital=capital
        )
        bucket_results[b_name] = (tdf, eqdf)
        
        n_trades = len(tdf)
        if n_trades > 0:
            wins = tdf[tdf['PnL ($)'] > 0]
            win_rate = (len(wins) / n_trades) * 100.0
            avg_ret = tdf['Return %'].mean()
            tot_pnl = tdf['PnL ($)'].sum()
            avg_days = tdf['Hold Days'].mean()
            
            gross_p = tdf.loc[tdf['PnL ($)'] > 0, 'PnL ($)'].sum()
            gross_l = abs(tdf.loc[tdf['PnL ($)'] < 0, 'PnL ($)'].sum())
            pf = (gross_p / gross_l) if gross_l > 0 else (np.inf if gross_p > 0 else 0.0)
            
            # Max Drawdown
            roll_max = eqdf['Equity'].cummax()
            dd = (eqdf['Equity'] - roll_max) / roll_max
            mdd = dd.min() * 100.0
        else:
            win_rate = avg_ret = tot_pnl = pf = mdd = avg_days = 0.0

        matrix_rows.append({
            "2-Day CRSI Bucket": b_name,
            "Total Trades": n_trades,
            "Win Rate %": round(win_rate, 1),
            "Avg Return %": round(avg_ret, 2),
            "Profit Factor": round(pf, 2) if pf != np.inf else "∞",
            "Total PnL ($)": round(tot_pnl, 2),
            "Max Drawdown %": round(mdd, 1),
            "Avg Hold (Days)": round(avg_days, 1)
        })
        
    summary_table = pd.DataFrame(matrix_rows)
    
    # Render table with styled win rate highlights
    st.dataframe(
        summary_table.style.format({
            "Win Rate %": "{:.1f}%",
            "Avg Return %": "{:.2f}%",
            "Total PnL ($)": "${:,.2f}",
            "Max Drawdown %": "{:.1f}%",
            "Avg Hold (Days)": "{:.1f}"
        }),
        use_container_width=True
    )
    
    # Comparative Bar Charts
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=summary_table["2-Day CRSI Bucket"],
        y=summary_table["Win Rate %"],
        name="Win Rate %",
        marker_color="#2962FF"
    ))
    fig_bar.add_trace(go.Bar(
        x=summary_table["2-Day CRSI Bucket"],
        y=summary_table["Avg Return %"],
        name="Avg Return % per Trade",
        yaxis="y2",
        marker_color="#00C805"
    ))
    fig_bar.update_layout(
        title="Win Rate and Average Trade Return by 2-Day CRSI Bucket",
        xaxis_title="2-Day CRSI Range",
        yaxis=dict(title="Win Rate (%)", side="left", range=[0, 100]),
        yaxis2=dict(title="Avg Return (%)", side="right", overlaying="y"),
        legend=dict(x=0.01, y=0.99),
        template="plotly_white",
        barmode="group",
        height=450
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ------------------------------------------
# TAB 2: DETAILED SINGLE BUCKET
# ------------------------------------------
with tab2:
    selected_bucket = st.selectbox("Select Bucket to Inspect:", list(BUCKET_RANGES.keys()), index=0)
    tdf_sel, eqdf_sel = bucket_results[selected_bucket]
    
    if tdf_sel.empty:
        st.warning(f"No trades were triggered for bucket '{selected_bucket}' with current parameters.")
    else:
        wins = tdf_sel[tdf_sel['PnL ($)'] > 0]
        losses = tdf_sel[tdf_sel['PnL ($)'] < 0]
        win_rate = (len(wins) / len(tdf_sel)) * 100.0
        tot_return = ((eqdf_sel['Equity'].iloc[-1] / capital) - 1) * 100.0
        
        gross_p = wins['PnL ($)'].sum()
        gross_l = abs(losses['PnL ($)'].sum())
        pf = (gross_p / gross_l) if gross_l > 0 else np.nan
        
        roll_max = eqdf_sel['Equity'].cummax()
        dd = (eqdf_sel['Equity'] - roll_max) / roll_max
        max_dd = dd.min() * 100.0
        
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Trades", len(tdf_sel))
        m2.metric("Win Rate", f"{win_rate:.1f}%")
        m3.metric("Total Return", f"{tot_return:.1f}%")
        m4.metric("Profit Factor", f"{pf:.2f}" if not np.isnan(pf) else "N/A")
        m5.metric("Max Drawdown", f"{max_dd:.1f}%")
        
        # Equity Curve
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=eqdf_sel.index,
            y=eqdf_sel['Equity'],
            mode='lines',
            line=dict(color='#00C805', width=2),
            name="Portfolio Value"
        ))
        fig_eq.update_layout(
            title=f"Equity Curve: Bucket [{selected_bucket}] on {ticker}",
            xaxis_title="Date",
            yaxis_title="Portfolio Equity ($)",
            template="plotly_white",
            hovermode="x unified",
            height=400
        )
        st.plotly_chart(fig_eq, use_container_width=True)
        
        # Trade Log
        st.subheader("Executed Trades")
        st.dataframe(tdf_sel, use_container_width=True)
