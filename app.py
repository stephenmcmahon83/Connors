import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import urllib.request

st.set_page_config(
    page_title="ConnorsRSI 2-Day Bucket Backtester & Screener",
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
    """Computes ConnorsRSI(3,2,100), its 2-day SMA, and rolling X-day highs."""
    df = df.copy()
    rsi_p = calculate_rsi(df['Close'], 3)
    rsi_s = calculate_rsi(calculate_streak(df['Close']), 2)
    pr = calculate_percent_rank(df['Close'], 100)
    
    df['CRSI'] = (rsi_p + rsi_s + pr) / 3.0
    df['CRSI_2D'] = df['CRSI'].rolling(2).mean()
    df['X_Day_High'] = df['High'].shift(1).rolling(x_days).max()
    return df

# ==========================================
# 2. BUCKET DEFINITIONS & BACKTEST LOGIC
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
    signal = (df['CRSI_2D'] >= low_b) & (df['CRSI_2D'] <= high_b)
    
    for i in range(1, len(df)):
        curr_date = dates[i]
        curr_close = df['Close'].iloc[i]
        curr_open = df['Open'].iloc[i]
        curr_high = df['High'].iloc[i]
        curr_crsi_2d = df['CRSI_2D'].iloc[i]
        x_high = df['X_Day_High'].iloc[i]
        
        # 1. Manage Exits
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

        # 2. Manage Entries
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
# 3. UNIVERSES & CACHED LOADERS
# ==========================================

MAJOR_ETFS = [
    "SPY", "QQQ", "IWM", "DIA", "SMH", "SOXX", "XLF", "XLE", "XLK", "XLV", "XLI", 
    "XLP", "XLU", "XLY", "XLC", "XLB", "XBI", "ARKK", "GLD", "SLV", "TLT", "HYG", 
    "EEM", "EFA", "FXI", "KWEB", "VNQ", "TQQQ", "SQQQ", "SOXL", "SPXL", "UVXY"
]

TOP_ADRS = [
    "TSM", "BABA", "ASML", "NVO", "AZN", "SAP", "TM", "HDB", "SE", "PDD", "MELI", 
    "NIO", "BIDU", "JD", "INFY", "BNS", "BP", "SHEL", "RIO", "VALE", "SNY", "GSK", 
    "SAN", "DEO", "UL", "ABB", "RELX", "ARGX", "SONY", "MUFG"
]

RUSSELL_2000_LIQUID = [
    "PLTR", "HOOD", "AFRM", "SOFI", "RBLX", "COIN", "DKNG", "SNAP", "DASH", "PINS",
    "CRWD", "NET", "SNOW", "PANW", "ZS", "MDB", "DDOG", "TTD", "ENPH", "CELH",
    "SYM", "PATH", "IONQ", "SMCI", "ARM", "RIVN", "LCID", "MARA", "RIOT", "CLSK",
    "APP", "DUOL", "SAIA", "ELF", "FSLR", "ONTO", "POWI", "EXAS", "ACLX", "HALO",
    "AXSM", "KNSL", "MEDP", "LNTH", "ALGM", "BOOT", "WING", "SHAK", "CAVA", "BLMN"
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
        return ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "BRK-B", "JPM", "V", "UNH"]

@st.cache_data(ttl=86400)
def get_nasdaq100_tickers() -> list:
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            tables = pd.read_html(resp)
        return sorted(tables[4]['Ticker'].str.replace('.', '-', regex=False).tolist())
    except Exception:
        return ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "COST", "AVGO", "TSLA", "AMD"]

# ==========================================
# 4. SIDEBAR SETTINGS
# ==========================================

st.sidebar.title("Strategy Rules")

selected_bucket_name = st.sidebar.selectbox(
    "Entry 2-Day CRSI Bucket",
    options=list(BUCKET_RANGES.keys()),
    index=0,
    help="Target bucket that triggers a buy signal."
)
target_lb, target_hb = BUCKET_RANGES[selected_bucket_name]

x_days = st.sidebar.slider("Exit Rule: X-Day High Lookback", min_value=1, max_value=20, value=5, step=1)

crsi_exit_thresh = st.sidebar.selectbox(
    "Exit Rule: 2-Day CRSI Exit Threshold (> N)",
    options=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
    index=7  # Default 70
)

exit_mode = st.sidebar.selectbox(
    "Exit Condition Logic",
    [
        "Either (Whichever occurs first)",
        "X-Day High Only",
        "2-Day CRSI Threshold Only",
        "Both Required"
    ]
)

entry_timing = st.sidebar.radio("Entry Execution", ["Next Open", "Signal Day Close"], horizontal=True)

# ==========================================
# 5. TABS: SCREENER & BACKTESTERS
# ==========================================

tab_screen, tab_matrix, tab_deep = st.tabs([
    "🔍 Live Market Screener",
    "📊 All Buckets Study (Larry Connors Matrix)",
    "📈 Single Bucket Deep Dive"
])

# ------------------------------------------
# TAB 1: LIVE MARKET SCREENER
# ------------------------------------------
with tab_screen:
    st.subheader("Market Screener: 2-Day ConnorsRSI, Signals & Open Positions")
    st.caption(
        f"Scanning for entry bucket **[{selected_bucket_name}]** | Exit: **{exit_mode}** "
        f"(X-High: {x_days}d, CRSI Exit: > {crsi_exit_thresh})"
    )
    
    col_u1, col_u2 = st.columns([3, 1])
    with col_u1:
        universes_selected = st.multiselect(
            "Select Universes to Scan:",
            ["S&P 500", "Nasdaq 100", "Russell 2000 Liquid", "Major US ETFs", "Top Global ADRs"],
            default=["Nasdaq 100", "Major US ETFs", "Top Global ADRs"]
        )
    with col_u2:
        max_scan = st.slider("Max Stocks to Scan", min_value=30, max_value=700, value=200, step=25)
        
    filter_actionable = st.checkbox("Show Only Actionable Stocks (Buy Signals, Open Positions, Exits)", value=False)
    run_scan_btn = st.button("🚀 Run Live Screener", type="primary")

    if run_scan_btn:
        tickers_to_scan = []
        if "S&P 500" in universes_selected:
            tickers_to_scan.extend(get_sp500_tickers())
        if "Nasdaq 100" in universes_selected:
            tickers_to_scan.extend(get_nasdaq100_tickers())
        if "Russell 2000 Liquid" in universes_selected:
            tickers_to_scan.extend(RUSSELL_2000_LIQUID)
        if "Major US ETFs" in universes_selected:
            tickers_to_scan.extend(MAJOR_ETFS)
        if "Top Global ADRs" in universes_selected:
            tickers_to_scan.extend(TOP_ADRS)
            
        tickers_to_scan = sorted(list(set(tickers_to_scan)))[:max_scan]
        
        st.info(f"Scanning {len(tickers_to_scan)} tickers in fast parallel batches...")
        progress_bar = st.progress(0.0)
        status_box = st.empty()
        
        results = []
        batch_size = 50
        num_batches = int(np.ceil(len(tickers_to_scan) / batch_size))
        
        for b in range(num_batches):
            batch = tickers_to_scan[b * batch_size : (b + 1) * batch_size]
            status_box.text(f"Fetching batch {b + 1}/{num_batches} ({len(batch)} tickers)...")
            
            try:
                raw_batch = yf.download(
                    tickers=batch,
                    period="1y",
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=True,
                    progress=False,
                    threads=True
                )
                
                for t in batch:
                    try:
                        if isinstance(raw_batch.columns, pd.MultiIndex):
                            if t in raw_batch.columns.get_level_values(0):
                                df_t = raw_batch[t].dropna(how='all')
                            elif t in raw_batch.columns.get_level_values(1):
                                df_t = raw_batch.xs(t, axis=1, level=1).dropna(how='all')
                            else:
                                continue
                        else:
                            df_t = raw_batch.dropna(how='all')
                            
                        if len(df_t) < 120:
                            continue
                            
                        calc_df = prepare_crsi_data(df_t, x_days=x_days)
                        
                        # Simulate the past 40 sessions to evaluate active trade state
                        sub_df = calc_df.iloc[-40:].copy()
                        in_trade = False
                        entry_price = 0.0
                        entry_date = None
                        
                        for idx_bar in range(1, len(sub_df) - 1):
                            bar_date = sub_df.index[idx_bar]
                            b_close = sub_df['Close'].iloc[idx_bar]
                            b_open = sub_df['Open'].iloc[idx_bar]
                            b_high = sub_df['High'].iloc[idx_bar]
                            b_crsi_2d = sub_df['CRSI_2D'].iloc[idx_bar]
                            b_x_high = sub_df['X_Day_High'].iloc[idx_bar]
                            
                            # Exit logic on historical bar
                            if in_trade:
                                hit_c = b_crsi_2d > crsi_exit_thresh
                                hit_xh = pd.notna(b_x_high) and (b_high >= b_x_high)
                                exit_now = False
                                if exit_mode == "Either (Whichever occurs first)" and (hit_c or hit_xh):
                                    exit_now = True
                                elif exit_mode == "Both Required" and (hit_c and hit_xh):
                                    exit_now = True
                                elif exit_mode == "X-Day High Only" and hit_xh:
                                    exit_now = True
                                elif exit_mode == "2-Day CRSI Threshold Only" and hit_c:
                                    exit_now = True
                                if exit_now:
                                    in_trade = False
                                    
                            # Entry logic on historical bar
                            if not in_trade:
                                prev_sig = (sub_df['CRSI_2D'].iloc[idx_bar - 1] >= target_lb) and (sub_df['CRSI_2D'].iloc[idx_bar - 1] <= target_hb)
                                curr_sig = (b_crsi_2d >= target_lb) and (b_crsi_2d <= target_hb)
                                
                                if entry_timing == "Next Open" and prev_sig:
                                    in_trade = True
                                    entry_price = b_open
                                    entry_date = bar_date
                                elif entry_timing == "Signal Day Close" and curr_sig:
                                    in_trade = True
                                    entry_price = b_close
                                    entry_date = bar_date
                        
                        # Latest bar analysis (Today)
                        latest = calc_df.iloc[-1]
                        latest_date = calc_df.index[-1]
                        curr_close = latest['Close']
                        curr_high = latest['High']
                        curr_crsi_2d = latest['CRSI_2D']
                        curr_crsi = latest['CRSI']
                        x_high_val = latest['X_Day_High']
                        
                        # Identify current 2-day CRSI bucket
                        current_bucket = "None"
                        for b_name, (b_min, b_max) in BUCKET_RANGES.items():
                            if b_min <= curr_crsi_2d <= b_max:
                                current_bucket = b_name
                                break
                                
                        status = "⚪ NEUTRAL"
                        action_needed = "No Action"
                        gain_pct = 0.0
                        days_held = 0
                        
                        # Evaluate latest bar against active position
                        if in_trade:
                            days_held = (latest_date - entry_date).days
                            gain_pct = ((curr_close / entry_price) - 1) * 100.0
                            
                            hit_c = curr_crsi_2d > crsi_exit_thresh
                            hit_xh = pd.notna(x_high_val) and (curr_high >= x_high_val)
                            
                            exit_triggered = False
                            reason = []
                            if hit_xh:
                                reason.append("Hit X-Day High")
                            if hit_c:
                                reason.append(f"CRSI 2D > {crsi_exit_thresh}")
                                
                            if exit_mode == "Either (Whichever occurs first)" and (hit_c or hit_xh):
                                exit_triggered = True
                            elif exit_mode == "Both Required" and (hit_c and hit_xh):
                                exit_triggered = True
                            elif exit_mode == "X-Day High Only" and hit_xh:
                                exit_triggered = True
                            elif exit_mode == "2-Day CRSI Threshold Only" and hit_c:
                                exit_triggered = True
                                
                            if exit_triggered:
                                status = "🚨 EXIT TOMORROW"
                                action_needed = f"Close at Open ({' & '.join(reason)})"
                            else:
                                status = "🟡 OPEN / HOLDING"
                                action_needed = f"Hold (Entry: ${entry_price:.2f})"
                        else:
                            # Check if today triggers an entry signal
                            is_signal = (target_lb <= curr_crsi_2d <= target_hb)
                            if is_signal:
                                status = "🟢 BUY SIGNAL"
                                action_needed = f"Buy Tomorrow ({entry_timing})"
                                
                        results.append({
                            "Ticker": t,
                            "Status": status,
                            "Action / Details": action_needed,
                            "2-Day CRSI": round(curr_crsi_2d, 2),
                            "Current Bucket": current_bucket,
                            "1-Day CRSI": round(curr_crsi, 2),
                            "Last Close": round(curr_close, 2),
                            "X-Day High": round(x_high_val, 2) if pd.notna(x_high_val) else np.nan,
                            "Unrealized PnL %": round(gain_pct, 2) if in_trade else np.nan,
                            "Days Held": days_held if in_trade else 0
                        })
                    except Exception:
                        continue
            except Exception:
                pass
            progress_bar.progress((b + 1) / num_batches)
            
        progress_bar.empty()
        status_box.empty()
        
        if results:
            res_df = pd.DataFrame(results)
            
            # Sort: actionable statuses first, then lowest 2-Day CRSI
            status_order = {"🚨 EXIT TOMORROW": 0, "🟢 BUY SIGNAL": 1, "🟡 OPEN / HOLDING": 2, "⚪ NEUTRAL": 3}
            res_df['order'] = res_df['Status'].map(status_order).fillna(4)
            res_df = res_df.sort_values(by=['order', '2-Day CRSI'], ascending=[True, True]).drop(columns=['order'])
            
            if filter_actionable:
                res_df = res_df[res_df['Status'] != "⚪ NEUTRAL"]
                
            n_buys = len(res_df[res_df['Status'] == "🟢 BUY SIGNAL"])
            n_exits = len(res_df[res_df['Status'] == "🚨 EXIT TOMORROW"])
            n_open = len(res_df[res_df['Status'] == "🟡 OPEN / HOLDING"])
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Scanned", len(results))
            col_m2.metric("Buy Signals Today", n_buys)
            col_m3.metric("Exits for Tomorrow", n_exits)
            col_m4.metric("Holding Positions", n_open)
            
            st.dataframe(
                res_df.style.format({
                    "2-Day CRSI": "{:.2f}",
                    "1-Day CRSI": "{:.2f}",
                    "Last Close": "${:.2f}",
                    "X-Day High": "${:.2f}",
                    "Unrealized PnL %": "{:+.2f}%"
                }, na_rep="-"),
                use_container_width=True,
                height=550
            )
        else:
            st.warning("No data retrieved for the selected parameters.")

# ------------------------------------------
# TAB 2: ALL BUCKETS MATRIX
# ------------------------------------------
with tab_matrix:
    bt_ticker = st.text_input("Matrix Benchmark Ticker", value="SPY", key="matrix_ticker").upper()
    start_bt = st.date_input("Start Date", value=datetime.today() - timedelta(days=10*365), key="matrix_date")
    
    if st.button("Run Larry Connors Bucket Study", type="primary"):
        with st.spinner(f"Computing 10-Bucket Matrix on {bt_ticker}..."):
            raw_bt = yf.download(bt_ticker, start=start_bt, interval="1d", progress=False, auto_adjust=True)
            if isinstance(raw_bt.columns, pd.MultiIndex):
                raw_bt.columns = raw_bt.columns.get_level_values(0)
                
            if len(raw_bt) >= 120:
                p_df = prepare_crsi_data(raw_bt, x_days=x_days)
                matrix_rows = []
                
                for b_name, (lb, hb) in BUCKET_RANGES.items():
                    tdf, eqdf = backtest_bucket(
                        df=p_df, low_b=lb, high_b=hb, exit_mode=exit_mode,
                        crsi_exit_thresh=crsi_exit_thresh, entry_timing=entry_timing
                    )
                    n_trades = len(tdf)
                    if n_trades > 0:
                        wins = tdf[tdf['PnL ($)'] > 0]
                        wr = (len(wins) / n_trades) * 100.0
                        avg_ret = tdf['Return %'].mean()
                        tot_pnl = tdf['PnL ($)'].sum()
                        avg_days = tdf['Hold Days'].mean()
                        gross_p = tdf.loc[tdf['PnL ($)'] > 0, 'PnL ($)'].sum()
                        gross_l = abs(tdf.loc[tdf['PnL ($)'] < 0, 'PnL ($)'].sum())
                        pf = (gross_p / gross_l) if gross_l > 0 else np.inf
                        roll_max = eqdf['Equity'].cummax()
                        dd = (eqdf['Equity'] - roll_max) / roll_max
                        mdd = dd.min() * 100.0
                    else:
                        wr = avg_ret = tot_pnl = pf = mdd = avg_days = 0.0

                    matrix_rows.append({
                        "2-Day CRSI Bucket": b_name,
                        "Total Trades": n_trades,
                        "Win Rate %": round(wr, 1),
                        "Avg Return %": round(avg_ret, 2),
                        "Profit Factor": round(pf, 2) if pf != np.inf else "∞",
                        "Total PnL ($)": round(tot_pnl, 2),
                        "Max Drawdown %": round(mdd, 1),
                        "Avg Hold (Days)": round(avg_days, 1)
                    })
                    
                sum_df = pd.DataFrame(matrix_rows)
                st.dataframe(
                    sum_df.style.format({
                        "Win Rate %": "{:.1f}%",
                        "Avg Return %": "{:.2f}%",
                        "Total PnL ($)": "${:,.2f}",
                        "Max Drawdown %": "{:.1f}%",
                        "Avg Hold (Days)": "{:.1f}"
                    }),
                    use_container_width=True
                )
                
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=sum_df["2-Day CRSI Bucket"], y=sum_df["Win Rate %"],
                    name="Win Rate %", marker_color="#2962FF"
                ))
                fig_bar.add_trace(go.Bar(
                    x=sum_df["2-Day CRSI Bucket"], y=sum_df["Avg Return %"],
                    name="Avg Return %", yaxis="y2", marker_color="#00C805"
                ))
                fig_bar.update_layout(
                    title="Win Rate & Average Trade Return across 2-Day CRSI Buckets",
                    xaxis_title="2-Day CRSI Band",
                    yaxis=dict(title="Win Rate (%)", side="left", range=[0, 100]),
                    yaxis2=dict(title="Avg Return (%)", side="right", overlaying="y"),
                    template="plotly_white", barmode="group", height=450
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.error("Insufficient historical data for backtesting.")

# ------------------------------------------
# TAB 3: SINGLE BUCKET DEEP DIVE
# ------------------------------------------
with tab_deep:
    c1, c2, c3 = st.columns(3)
    with c1:
        deep_ticker = st.text_input("Ticker Symbol", value="SPY", key="deep_ticker").upper()
    with c2:
        deep_start = st.date_input("Start Date", value=datetime.today() - timedelta(days=10*365), key="deep_date")
    with c3:
        deep_bucket = st.selectbox("Inspect Bucket", list(BUCKET_RANGES.keys()), index=0, key="deep_bucket")
        
    if st.button("Run Single Bucket Backtest", type="primary"):
        with st.spinner("Calculating performance..."):
            raw_single = yf.download(deep_ticker, start=deep_start, interval="1d", progress=False, auto_adjust=True)
            if isinstance(raw_single.columns, pd.MultiIndex):
                raw_single.columns = raw_single.columns.get_level_values(0)
                
            if len(raw_single) >= 120:
                p_df_single = prepare_crsi_data(raw_single, x_days=x_days)
                b_min, b_max = BUCKET_RANGES[deep_bucket]
                tdf_s, eqdf_s = backtest_bucket(
                    p_df_single, low_b=b_min, high_b=b_max, exit_mode=exit_mode,
                    crsi_exit_thresh=crsi_exit_thresh, entry_timing=entry_timing
                )
                
                if tdf_s.empty:
                    st.warning("No trades triggered for this bucket.")
                else:
                    wr = (len(tdf_s[tdf_s['PnL ($)'] > 0]) / len(tdf_s)) * 100.0
                    tot_ret = ((eqdf_s['Equity'].iloc[-1] / 10000.0) - 1) * 100.0
                    gp = tdf_s.loc[tdf_s['PnL ($)'] > 0, 'PnL ($)'].sum()
                    gl = abs(tdf_s.loc[tdf_s['PnL ($)'] < 0, 'PnL ($)'].sum())
                    pf = (gp / gl) if gl > 0 else np.nan
                    
                    roll_max = eqdf_s['Equity'].cummax()
                    mdd = ((eqdf_s['Equity'] - roll_max) / roll_max).min() * 100.0
                    
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Trades", len(tdf_s))
                    m2.metric("Win Rate", f"{wr:.1f}%")
                    m3.metric("Profit Factor", f"{pf:.2f}" if not np.isnan(pf) else "N/A")
                    m4.metric("Max Drawdown", f"{mdd:.1f}%")
                    
                    fig_eq = go.Figure()
                    fig_eq.add_trace(go.Scatter(
                        x=eqdf_s.index, y=eqdf_s['Equity'],
                        mode='lines', line=dict(color='#00C805', width=2)
                    ))
                    fig_eq.update_layout(
                        title=f"{deep_ticker} Equity Curve — Bucket [{deep_bucket}]",
                        xaxis_title="Date", yaxis_title="Equity ($)",
                        template="plotly_white", height=420
                    )
                    st.plotly_chart(fig_eq, use_container_width=True)
                    st.dataframe(tdf_s, use_container_width=True)
