import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import urllib.request

st.set_page_config(
    page_title="ConnorsRSI Universe Backtester & Screener",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 1. CORE TECHNICAL INDICATORS
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
    """Consecutive up/down close days counter."""
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
    """Computes RSI(3), ConnorsRSI(3,2,100), 2-Day CRSI, and X-Day Highs."""
    df = df.copy()
    df['RSI3'] = calculate_rsi(df['Close'], 3)
    rsi_streak = calculate_rsi(calculate_streak(df['Close']), 2)
    pr = calculate_percent_rank(df['Close'], 100)
    
    df['CRSI'] = (df['RSI3'] + rsi_streak + pr) / 3.0
    df['CRSI_2D'] = df['CRSI'].rolling(2).mean()
    df['X_Day_High'] = df['High'].shift(1).rolling(x_days).max()
    
    # Falling RSI conditions (today < yesterday)
    df['CRSI_Falling'] = df['CRSI'] < df['CRSI'].shift(1)
    df['RSI3_Falling'] = df['RSI3'] < df['RSI3'].shift(1)
    return df

# ==========================================
# 2. BUCKET DEFINITIONS & UNIVERSES
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

MAJOR_ETFS_20YR = [
    "SPY", "QQQ", "IWM", "DIA", "SMH", "XLF", "XLE", "XLK", "XLV", 
    "XLI", "XLP", "XLU", "XLY", "XLB", "GLD", "TLT", "EEM", "EFA"
]

MEGA_CAPS_20YR = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "JPM", "JNJ", "PG", "XOM", 
    "HD", "IBM", "CSCO", "INTC", "WMT", "KO", "DIS", "BA", "CAT", "MCD", "GE"
]

@st.cache_data(ttl=86400)
def get_sp500_tickers() -> list:
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            tables = pd.read_html(resp)
        return sorted(tables[0]['Symbol'].str.replace('.', '-', regex=False).tolist())
    except Exception:
        return MEGA_CAPS_20YR

@st.cache_data(ttl=86400)
def get_nasdaq100_tickers() -> list:
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            tables = pd.read_html(resp)
        return sorted(tables[4]['Ticker'].str.replace('.', '-', regex=False).tolist())
    except Exception:
        return ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "COST", "CSCO", "INTC", "ADBE", "QCOM"]

# ==========================================
# 3. 20-YEAR PORTFOLIO UNIVERSE ENGINE
# ==========================================

def run_universe_backtest(
    data_dict: dict,
    low_b: float,
    high_b: float,
    exit_mode: str,
    x_days: int,
    crsi_exit_thresh: float,
    require_falling: bool,
    falling_type: str,
    entry_timing: str,
    initial_capital: float,
    max_positions: int
):
    """Simulates a multi-asset portfolio with capital allocation and ranking."""
    # Find common date index
    all_dates = sorted(list(set().union(*[df.index.tolist() for df in data_dict.values()])))
    if len(all_dates) < 120:
        return pd.DataFrame(), pd.DataFrame()

    capital = initial_capital
    active_positions = {}  # ticker -> {'entry_price', 'entry_date', 'shares', 'cost'}
    trades = []
    portfolio_equity = []
    
    # Pre-determine falling column
    falling_col = 'CRSI_Falling' if falling_type == "ConnorsRSI (CRSI)" else 'RSI3_Falling'

    for i in range(1, len(all_dates)):
        curr_date = all_dates[i]
        prev_date = all_dates[i - 1]
        
        # 1. PROCESS EXITS FIRST FOR ACTIVE POSITIONS
        tickers_to_exit = []
        for ticker, pos in list(active_positions.items()):
            df = data_dict[ticker]
            if curr_date not in df.index:
                continue
                
            curr_bar = df.loc[curr_date]
            c_high = curr_bar['High']
            c_close = curr_bar['Close']
            c_open = curr_bar['Open']
            c_crsi_2d = curr_bar['CRSI_2D']
            x_high = curr_bar['X_Day_High']
            
            hit_crsi = c_crsi_2d > crsi_exit_thresh
            hit_x_high = pd.notna(x_high) and (c_high >= x_high)
            
            should_exit = False
            exit_price = c_close
            exit_reason = ""
            
            if exit_mode == "Either (Whichever occurs first)":
                if hit_x_high and hit_crsi:
                    exit_price = max(c_open, x_high)
                    exit_reason = "Both (X-High & CRSI)"
                    should_exit = True
                elif hit_x_high:
                    exit_price = max(c_open, x_high)
                    exit_reason = "X-Day High"
                    should_exit = True
                elif hit_crsi:
                    exit_price = c_close
                    exit_reason = f"CRSI 2D > {crsi_exit_thresh}"
                    should_exit = True
            elif exit_mode == "Both Required":
                if hit_x_high and hit_crsi:
                    exit_price = c_close
                    exit_reason = "Both Met"
                    should_exit = True
            elif exit_mode == "X-Day High Only" and hit_x_high:
                exit_price = max(c_open, x_high)
                exit_reason = "X-Day High"
                should_exit = True
            elif exit_mode == "2-Day CRSI Threshold Only" and hit_crsi:
                exit_price = c_close
                exit_reason = f"CRSI 2D > {crsi_exit_thresh}"
                should_exit = True
                
            if should_exit:
                proceeds = pos['shares'] * exit_price
                pnl = proceeds - pos['cost']
                ret_pct = ((exit_price / pos['entry_price']) - 1) * 100.0
                capital += proceeds
                
                trades.append({
                    'Ticker': ticker,
                    'Entry Date': pos['entry_date'],
                    'Entry Price': round(pos['entry_price'], 2),
                    'Exit Date': curr_date,
                    'Exit Price': round(exit_price, 2),
                    'Exit Reason': exit_reason,
                    'Hold Days': (curr_date - pos['entry_date']).days,
                    'Return %': round(ret_pct, 2),
                    'PnL ($)': round(pnl, 2)
                })
                tickers_to_exit.append(ticker)
                
        for t in tickers_to_exit:
            del active_positions[t]
            
        # 2. EVALUATE ENTRY SIGNALS
        available_slots = max_positions - len(active_positions)
        if available_slots > 0:
            candidates = []
            
            for ticker, df in data_dict.items():
                if ticker in active_positions:
                    continue
                    
                check_date = prev_date if entry_timing == "Next Open" else curr_date
                if check_date not in df.index:
                    continue
                    
                bar = df.loc[check_date]
                crsi_2d = bar['CRSI_2D']
                
                in_bucket = (crsi_2d >= low_b) and (crsi_2d <= high_b)
                passes_falling = True
                if require_falling:
                    passes_falling = bool(bar[falling_col])
                    
                if in_bucket and passes_falling:
                    candidates.append({
                        'Ticker': ticker,
                        'CRSI_2D': crsi_2d,
                        'Execution Bar Date': curr_date
                    })
                    
            if candidates:
                # Rank candidates by lowest 2-Day CRSI (most oversold first)
                candidates = sorted(candidates, key=lambda c: c['CRSI_2D'])[:available_slots]
                
                # Portfolio value estimation for slot allocation
                current_unrealized = sum(
                    active_positions[t]['shares'] * data_dict[t].loc[curr_date]['Close']
                    for t in active_positions if curr_date in data_dict[t].index
                )
                total_equity = capital + current_unrealized
                target_slot_size = total_equity / max_positions
                
                for cand in candidates:
                    t = cand['Ticker']
                    df = data_dict[t]
                    if curr_date not in df.index:
                        continue
                        
                    buy_price = df.loc[curr_date]['Open'] if entry_timing == "Next Open" else df.loc[curr_date]['Close']
                    if buy_price <= 0:
                        continue
                        
                    position_capital = min(capital, target_slot_size)
                    shares = int(position_capital // buy_price)
                    
                    if shares > 0:
                        cost = shares * buy_price
                        capital -= cost
                        active_positions[t] = {
                            'entry_price': buy_price,
                            'entry_date': curr_date,
                            'shares': shares,
                            'cost': cost
                        }
        
        # 3. RECORD DAILY PORTFOLIO VALUE
        open_val = 0.0
        for t, pos in active_positions.items():
            if curr_date in data_dict[t].index:
                open_val += pos['shares'] * data_dict[t].loc[curr_date]['Close']
            else:
                open_val += pos['cost']
                
        daily_equity = capital + open_val
        portfolio_equity.append({'Date': curr_date, 'Equity': daily_equity, 'Open Positions': len(active_positions)})
        
    equity_df = pd.DataFrame(portfolio_equity).set_index('Date') if portfolio_equity else pd.DataFrame()
    trades_df = pd.DataFrame(trades)
    return trades_df, equity_df

# ==========================================
# 4. SIDEBAR CONFIGURATION
# ==========================================

st.sidebar.title("Strategy Rules & Filter")

selected_bucket = st.sidebar.selectbox(
    "1. 2-Day ConnorsRSI Bucket",
    options=list(BUCKET_RANGES.keys()),
    index=0
)
target_lb, target_hb = BUCKET_RANGES[selected_bucket]

# FALLING RSI FILTER TOGGLE
st.sidebar.markdown("---")
st.sidebar.subheader("Momentum / Declining Filter")
require_falling_rsi = st.sidebar.checkbox(
    "Require Latest RSI < Prior Day RSI",
    value=False,
    help="When enabled, signals are only valid if today's RSI reading is strictly lower than yesterday's reading (downward momentum)."
)

falling_type = st.sidebar.radio(
    "RSI Metric for Falling Filter",
    ["ConnorsRSI (CRSI)", "Price RSI (RSI-3)"],
    index=0,
    disabled=not require_falling_rsi
)

st.sidebar.markdown("---")
st.sidebar.subheader("Exit Rules")
x_days = st.sidebar.slider("Exit: Prior X-Day High", min_value=1, max_value=20, value=5, step=1)
crsi_exit_thresh = st.sidebar.selectbox("Exit: 2-Day CRSI Threshold (> N)", [0, 10, 20, 30, 40, 50, 60, 70, 80, 90], index=7)
exit_mode = st.sidebar.selectbox(
    "Exit Condition Logic",
    ["Either (Whichever occurs first)", "X-Day High Only", "2-Day CRSI Threshold Only", "Both Required"]
)
entry_timing = st.sidebar.radio("Entry Execution", ["Next Open", "Signal Day Close"], horizontal=True)

# ==========================================
# 5. TABS: UNIVERSE BACKTEST, MATRIX, SCREENER
# ==========================================

tab_universe, tab_matrix, tab_screener = st.tabs([
    "🌐 20-Year Universe Backtester",
    "📊 All-Buckets Matrix (Single Ticker)",
    "🔍 Live Market Screener"
])

# ------------------------------------------
# TAB 1: 20-YEAR UNIVERSE PORTFOLIO BACKTEST
# ------------------------------------------
with tab_universe:
    st.subheader("20-Year Multi-Asset Portfolio Backtest")
    st.caption("Simulates a complete portfolio trading across a basket of assets with slot allocation, max open positions, and ranking.")
    
    u_col1, u_col2, u_col3 = st.columns([2, 1, 1])
    with u_col1:
        universe_choice = st.selectbox(
            "Select Universe Basket:",
            ["Major Liquid ETFs (20+ Years History)", "Mega-Cap US Equities (20+ Years History)", "Nasdaq 100", "S&P 500 (Top 100 Liquid)", "Custom Ticker Basket"]
        )
    with u_col2:
        start_year = st.selectbox("Start Year (Up to ~20 Years)", options=list(range(2004, 2025)), index=2)  # Default 2006
    with u_col3:
        initial_cap = st.number_input("Starting Capital ($)", value=25000.0, step=5000.0)

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        max_pos = st.slider("Max Concurrent Open Positions", min_value=1, max_value=20, value=5, step=1)
    with p_col2:
        custom_input = st.text_input("Custom Ticker Basket (if selected):", value="SPY, QQQ, DIA, IWM, SMH, XLF, XLE, GLD, TLT")

    if st.button("🚀 Run 20-Year Universe Backtest", type="primary"):
        # Resolve tickers
        if universe_choice == "Major Liquid ETFs (20+ Years History)":
            chosen_tickers = MAJOR_ETFS_20YR
        elif universe_choice == "Mega-Cap US Equities (20+ Years History)":
            chosen_tickers = MEGA_CAPS_20YR
        elif universe_choice == "Nasdaq 100":
            chosen_tickers = get_nasdaq100_tickers()[:60]
        elif universe_choice == "S&P 500 (Top 100 Liquid)":
            chosen_tickers = get_sp500_tickers()[:100]
        else:
            chosen_tickers = [t.strip().upper() for t in custom_input.split(",") if t.strip()]

        start_dt = datetime(start_year, 1, 1)
        st.info(f"Downloading historical data since {start_year} for {len(chosen_tickers)} symbols...")
        
        with st.spinner("Downloading and processing indicators across the universe..."):
            batch_df = yf.download(
                tickers=chosen_tickers,
                start=start_dt,
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True
            )
            
            data_dict = {}
            for t in chosen_tickers:
                try:
                    if isinstance(batch_df.columns, pd.MultiIndex):
                        if t in batch_df.columns.get_level_values(0):
                            df_t = batch_df[t].dropna(how='all')
                        elif t in batch_df.columns.get_level_values(1):
                            df_t = batch_df.xs(t, axis=1, level=1).dropna(how='all')
                        else:
                            continue
                    else:
                        df_t = batch_df.dropna(how='all')
                    
                    if len(df_t) >= 120:
                        data_dict[t] = prepare_crsi_data(df_t, x_days=x_days)
                except Exception:
                    continue
                    
        if len(data_dict) == 0:
            st.error("No sufficient data loaded for the chosen tickers.")
        else:
            with st.spinner("Simulating multi-position portfolio backtest..."):
                trades_df, equity_df = run_universe_backtest(
                    data_dict=data_dict,
                    low_b=target_lb,
                    high_b=target_hb,
                    exit_mode=exit_mode,
                    x_days=x_days,
                    crsi_exit_thresh=crsi_exit_thresh,
                    require_falling=require_falling_rsi,
                    falling_type=falling_type,
                    entry_timing=entry_timing,
                    initial_capital=initial_cap,
                    max_positions=max_pos
                )
                
            if trades_df.empty:
                st.warning("No trades triggered across this universe with current parameters.")
            else:
                n_trades = len(trades_df)
                wins = trades_df[trades_df['PnL ($)'] > 0]
                losses = trades_df[trades_df['PnL ($)'] < 0]
                win_rate = (len(wins) / n_trades) * 100.0
                
                final_eq = equity_df['Equity'].iloc[-1]
                tot_ret = ((final_eq / initial_cap) - 1) * 100.0
                years = max((equity_df.index[-1] - equity_df.index[0]).days / 365.25, 1.0)
                cagr = (((final_eq / initial_cap) ** (1 / years)) - 1) * 100.0
                
                gp = wins['PnL ($)'].sum()
                gl = abs(losses['PnL ($)'].sum())
                pf = (gp / gl) if gl > 0 else np.nan
                
                roll_max = equity_df['Equity'].cummax()
                drawdown = (equity_df['Equity'] - roll_max) / roll_max
                max_dd = drawdown.min() * 100.0
                
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Total Trades", n_trades)
                m2.metric("Win Rate", f"{win_rate:.1f}%")
                m3.metric("Total Return", f"{tot_ret:.1f}%")
                m4.metric("CAGR", f"{cagr:.1f}%")
                m5.metric("Profit Factor", f"{pf:.2f}" if not np.isnan(pf) else "N/A")
                m6.metric("Max Drawdown", f"{max_dd:.1f}%")
                
                # Portfolio Equity Curve
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=equity_df.index,
                    y=equity_df['Equity'],
                    mode='lines',
                    name='Portfolio Equity',
                    line=dict(color='#00C805', width=2)
                ))
                fig.update_layout(
                    title=f"Portfolio Equity Curve ({start_year} – Present | {universe_choice})",
                    xaxis_title="Date",
                    yaxis_title="Total Equity ($)",
                    template="plotly_white",
                    hovermode="x unified",
                    height=450
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Breakdown by Ticker & Trade Log
                st.subheader("Performance Breakdown by Ticker")
                ticker_perf = trades_df.groupby('Ticker').agg(
                    Trades=('PnL ($)', 'count'),
                    Win_Rate=('PnL ($)', lambda s: (s > 0).mean() * 100),
                    Total_PnL=('PnL ($)', 'sum'),
                    Avg_Return=('Return %', 'mean')
                ).reset_index().sort_values(by='Total_PnL', ascending=False)
                
                st.dataframe(ticker_perf.style.format({
                    "Win_Rate": "{:.1f}%",
                    "Total_PnL": "${:,.2f}",
                    "Avg_Return": "{:.2f}%"
                }), use_container_width=True)
                
                st.subheader("Executed Trade Log")
                st.dataframe(trades_df, use_container_width=True)

# ------------------------------------------
# TAB 2: ALL BUCKETS MATRIX (SINGLE TICKER)
# ------------------------------------------
with tab_matrix:
    st.subheader("Larry Connors 10-Bucket Comparative Matrix")
    
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        matrix_ticker = st.text_input("Benchmark Ticker", value="SPY").upper()
    with b_col2:
        m_start_year = st.selectbox("Historical Horizon", [2004, 2006, 2010, 2015, 2020], index=1)
        
    if st.button("Run Bucket Matrix", type="primary"):
        with st.spinner(f"Analyzing all 10 buckets for {matrix_ticker}..."):
            raw_single = yf.download(matrix_ticker, start=datetime(m_start_year, 1, 1), progress=False, auto_adjust=True)
            if isinstance(raw_single.columns, pd.MultiIndex):
                raw_single.columns = raw_single.columns.get_level_values(0)
                
            if len(raw_single) >= 120:
                p_df = prepare_crsi_data(raw_single, x_days=x_days)
                rows = []
                
                for b_name, (lb, hb) in BUCKET_RANGES.items():
                    # Single ticker run
                    trades_b = []
                    in_trade = False
                    entry_p = 0.0
                    entry_d = None
                    capital_b = 10000.0
                    
                    sig = (p_df['CRSI_2D'] >= lb) & (p_df['CRSI_2D'] <= hb)
                    if require_falling_rsi:
                        f_col = 'CRSI_Falling' if falling_type == "ConnorsRSI (CRSI)" else 'RSI3_Falling'
                        sig = sig & p_df[f_col]
                        
                    for i in range(1, len(p_df)):
                        c_bar = p_df.iloc[i]
                        
                        if in_trade:
                            h_crsi = c_bar['CRSI_2D'] > crsi_exit_thresh
                            h_xh = pd.notna(c_bar['X_Day_High']) and (c_bar['High'] >= c_bar['X_Day_High'])
                            
                            exit_flag = False
                            exit_price = c_bar['Close']
                            if exit_mode == "Either (Whichever occurs first)" and (h_crsi or h_xh):
                                exit_flag = True
                                exit_price = max(c_bar['Open'], c_bar['X_Day_High']) if h_xh else c_bar['Close']
                            elif exit_mode == "Both Required" and (h_crsi and h_xh):
                                exit_flag = True
                            elif exit_mode == "X-Day High Only" and h_xh:
                                exit_flag = True
                                exit_price = max(c_bar['Open'], c_bar['X_Day_High'])
                            elif exit_mode == "2-Day CRSI Threshold Only" and h_crsi:
                                exit_flag = True
                                
                            if exit_flag:
                                ret_pct = ((exit_price / entry_p) - 1) * 100
                                pnl = (ret_pct / 100) * capital_b
                                capital_b += pnl
                                trades_b.append({'Return': ret_pct, 'PnL': pnl, 'Days': (p_df.index[i] - entry_d).days})
                                in_trade = False
                                
                        if not in_trade:
                            trigger = sig.iloc[i - 1] if entry_timing == "Next Open" else sig.iloc[i]
                            if trigger:
                                entry_p = c_bar['Open'] if entry_timing == "Next Open" else c_bar['Close']
                                entry_d = p_df.index[i]
                                in_trade = True
                                
                    n_b = len(trades_b)
                    if n_b > 0:
                        tdf_b = pd.DataFrame(trades_b)
                        wr = (len(tdf_b[tdf_b['PnL'] > 0]) / n_b) * 100
                        avg_r = tdf_b['Return'].mean()
                        tot_p = tdf_b['PnL'].sum()
                        avg_d = tdf_b['Days'].mean()
                    else:
                        wr = avg_r = tot_p = avg_d = 0.0
                        
                    rows.append({
                        "2-Day CRSI Bucket": b_name,
                        "Total Trades": n_b,
                        "Win Rate %": round(wr, 1),
                        "Avg Return %": round(avg_r, 2),
                        "Total PnL ($)": round(tot_p, 2),
                        "Avg Hold (Days)": round(avg_d, 1)
                    })
                    
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ------------------------------------------
# TAB 3: LIVE MARKET SCREENER
# ------------------------------------------
with tab_screener:
    st.subheader("Market Screener: 2-Day CRSI & Falling RSI Validator")
    st.caption(f"Scanning target bucket **[{selected_bucket}]** | Falling RSI Filter: **{require_falling_rsi}** ({falling_type if require_falling_rsi else 'Off'})")
    
    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        screener_univ = st.multiselect(
            "Universes:",
            ["Major US ETFs", "Mega-Cap US Equities", "Nasdaq 100", "S&P 500 (Top 100)"],
            default=["Major US ETFs", "Mega-Cap US Equities"]
        )
    with col_s2:
        screener_cap = st.slider("Limit Tickers", 20, 200, 60, step=20)
        
    only_signals = st.checkbox("Show Only Buy Signals & Exits", value=False)
    
    if st.button("🚀 Run Live Screener", type="primary"):
        s_tickers = []
        if "Major US ETFs" in screener_univ:
            s_tickers.extend(MAJOR_ETFS_20YR)
        if "Mega-Cap US Equities" in screener_univ:
            s_tickers.extend(MEGA_CAPS_20YR)
        if "Nasdaq 100" in screener_univ:
            s_tickers.extend(get_nasdaq100_tickers())
        if "S&P 500 (Top 100)" in screener_univ:
            s_tickers.extend(get_sp500_tickers()[:100])
            
        s_tickers = sorted(list(set(s_tickers)))[:screener_cap]
        
        with st.spinner(f"Scanning {len(s_tickers)} assets..."):
            s_data = yf.download(s_tickers, period="1y", interval="1d", group_by="ticker", auto_adjust=True, progress=False, threads=True)
            results = []
            
            f_col = 'CRSI_Falling' if falling_type == "ConnorsRSI (CRSI)" else 'RSI3_Falling'
            
            for t in s_tickers:
                try:
                    if isinstance(s_data.columns, pd.MultiIndex):
                        if t in s_data.columns.get_level_values(0):
                            df_t = s_data[t].dropna(how='all')
                        else:
                            continue
                    else:
                        df_t = s_data.dropna(how='all')
                        
                    if len(df_t) < 120:
                        continue
                        
                    df_calc = prepare_crsi_data(df_t, x_days=x_days)
                    latest = df_calc.iloc[-1]
                    prev = df_calc.iloc[-2]
                    
                    c_crsi_2d = latest['CRSI_2D']
                    c_crsi = latest['CRSI']
                    c_rsi3 = latest['RSI3']
                    is_falling = bool(latest[f_col])
                    
                    in_bucket = (target_lb <= c_crsi_2d <= target_hb)
                    passes_filter = (not require_falling_rsi) or is_falling
                    
                    status = "⚪ NEUTRAL"
                    if in_bucket and passes_filter:
                        status = "🟢 BUY SIGNAL"
                    elif c_crsi_2d > crsi_exit_thresh or (pd.notna(latest['X_Day_High']) and latest['High'] >= latest['X_Day_High']):
                        status = "🚨 EXIT MET"
                        
                    results.append({
                        "Ticker": t,
                        "Status": status,
                        "2-Day CRSI": round(c_crsi_2d, 2),
                        "CRSI Today": round(c_crsi, 2),
                        "CRSI Prev": round(prev['CRSI'], 2),
                        "RSI(3) Today": round(c_rsi3, 2),
                        "RSI Falling?": "✅ Yes" if is_falling else "❌ No",
                        "Close": round(latest['Close'], 2),
                        "X-Day High": round(latest['X_Day_High'], 2) if pd.notna(latest['X_Day_High']) else np.nan
                    })
                except Exception:
                    continue
                    
            if results:
                res_df = pd.DataFrame(results).sort_values(by="2-Day CRSI")
                if only_signals:
                    res_df = res_df[res_df['Status'] != "⚪ NEUTRAL"]
                st.dataframe(res_df, use_container_width=True)
            else:
                st.warning("No data retrieved.")
