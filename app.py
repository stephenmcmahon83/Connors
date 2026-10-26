import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import urllib.request

st.set_page_config(
    page_title="ConnorsRSI Multi-Bucket Backtester & Screener",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 1. TECHNICAL INDICATORS
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
    """Calculates RSI(3), ConnorsRSI(3,2,100), 2-Day CRSI, and X-Day Highs."""
    df = df.copy()
    df['RSI3'] = calculate_rsi(df['Close'], 3)
    rsi_streak = calculate_rsi(calculate_streak(df['Close']), 2)
    pr = calculate_percent_rank(df['Close'], 100)
    
    df['CRSI'] = (df['RSI3'] + rsi_streak + pr) / 3.0
    df['CRSI_2D'] = df['CRSI'].rolling(2).mean()
    df['X_Day_High'] = df['High'].shift(1).rolling(x_days).max()
    
    # Declining / Falling RSI flags (Today < Yesterday)
    df['CRSI_Falling'] = df['CRSI'] < df['CRSI'].shift(1)
    df['RSI3_Falling'] = df['RSI3'] < df['RSI3'].shift(1)
    return df

# ==========================================
# 2. BUCKET UTILITIES
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

def get_bucket_mask(series: pd.Series, selected_buckets: list) -> pd.Series:
    """Returns a boolean mask that is True if values fall into ANY selected bucket."""
    if not selected_buckets:
        return pd.Series(False, index=series.index)
    mask = pd.Series(False, index=series.index)
    for b in selected_buckets:
        lb, hb = BUCKET_RANGES[b]
        mask = mask | ((series >= lb) & (series <= hb))
    return mask

def check_value_in_buckets(val: float, selected_buckets: list) -> bool:
    """Checks whether a single float value falls into ANY selected bucket."""
    if not selected_buckets or pd.isna(val):
        return False
    for b in selected_buckets:
        lb, hb = BUCKET_RANGES[b]
        if lb <= val <= hb:
            return True
    return False

# ==========================================
# 3. COMPLETE UNCAPPED UNIVERSES
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

RUSSELL_REPRESENTATIVE = [
    "PLTR", "HOOD", "AFRM", "SOFI", "RBLX", "COIN", "DKNG", "SNAP", "DASH", "PINS",
    "CRWD", "NET", "SNOW", "PANW", "ZS", "MDB", "DDOG", "TTD", "ENPH", "CELH",
    "SYM", "PATH", "IONQ", "SMCI", "ARM", "RIVN", "LCID", "MARA", "RIOT", "CLSK",
    "APP", "DUOL", "SAIA", "ELF", "FSLR", "ONTO", "POWI", "EXAS", "ACLX", "HALO",
    "AXSM", "KNSL", "MEDP", "LNTH", "ALGM", "BOOT", "WING", "SHAK", "CAVA", "BLMN"
]

@st.cache_data(ttl=86400)
def get_sp500_tickers() -> list:
    """Fetches the complete S&P 500 universe."""
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
    """Fetches the complete Nasdaq 100 universe."""
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            tables = pd.read_html(resp)
        return sorted(tables[4]['Ticker'].str.replace('.', '-', regex=False).tolist())
    except Exception:
        return ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "COST", "AVGO", "TSLA", "AMD"]

# ==========================================
# 4. MULTI-ASSET PORTFOLIO BACKTEST ENGINE
# ==========================================

def run_universe_backtest(
    data_dict: dict,
    selected_buckets: list,
    exit_mode: str,
    x_days: int,
    crsi_exit_thresh: float,
    require_falling: bool,
    falling_type: str,
    entry_timing: str,
    initial_capital: float,
    max_positions: int
):
    all_dates = sorted(list(set().union(*[df.index.tolist() for df in data_dict.values()])))
    if len(all_dates) < 120:
        return pd.DataFrame(), pd.DataFrame()

    capital = initial_capital
    active_positions = {}
    trades = []
    portfolio_equity = []
    
    falling_col = 'CRSI_Falling' if falling_type == "ConnorsRSI (CRSI)" else 'RSI3_Falling'

    for i in range(1, len(all_dates)):
        curr_date = all_dates[i]
        prev_date = all_dates[i - 1]
        
        # 1. PROCESS EXITS
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
            
        # 2. PROCESS ENTRIES ACROSS MULTIPLE SELECTED BUCKETS
        available_slots = max_positions - len(active_positions)
        if available_slots > 0 and selected_buckets:
            candidates = []
            
            for ticker, df in data_dict.items():
                if ticker in active_positions:
                    continue
                    
                check_date = prev_date if entry_timing == "Next Open" else curr_date
                if check_date not in df.index:
                    continue
                    
                bar = df.loc[check_date]
                crsi_2d = bar['CRSI_2D']
                
                # Check multi-bucket membership
                in_bucket = check_value_in_buckets(crsi_2d, selected_buckets)
                passes_falling = True
                if require_falling:
                    passes_falling = bool(bar[falling_col])
                    
                if in_bucket and passes_falling:
                    candidates.append({
                        'Ticker': ticker,
                        'CRSI_2D': crsi_2d,
                        'Date': curr_date
                    })
                    
            if candidates:
                # Rank candidates by lowest 2-Day CRSI first
                candidates = sorted(candidates, key=lambda c: c['CRSI_2D'])[:available_slots]
                
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
        
        # 3. DAILY PORTFOLIO TRACKING
        open_val = sum(
            pos['shares'] * (data_dict[t].loc[curr_date]['Close'] if curr_date in data_dict[t].index else pos['entry_price'])
            for t, pos in active_positions.items()
        )
        portfolio_equity.append({'Date': curr_date, 'Equity': capital + open_val})
        
    equity_df = pd.DataFrame(portfolio_equity).set_index('Date') if portfolio_equity else pd.DataFrame()
    trades_df = pd.DataFrame(trades)
    return trades_df, equity_df

# ==========================================
# 5. SIDEBAR: MULTI-BUCKET SELECTION & RULES
# ==========================================

st.sidebar.title("Strategy Rules")

# MULTI-BUCKET SELECTION
selected_buckets = st.sidebar.multiselect(
    "1. Entry 2-Day ConnorsRSI Buckets (Select 1 or more)",
    options=list(BUCKET_RANGES.keys()),
    default=["0 - 10", "11 - 20"],
    help="Positions are entered when the 2-day CRSI falls within ANY of the selected buckets."
)

if not selected_buckets:
    st.sidebar.warning("⚠️ Please select at least one bucket to generate entries.")

# FALLING RSI FILTER
st.sidebar.markdown("---")
st.sidebar.subheader("Momentum / Declining Filter")
require_falling_rsi = st.sidebar.checkbox(
    "Require Latest RSI < Prior Day RSI",
    value=False,
    help="When checked, an entry is only valid if today's RSI reading is strictly lower than yesterday's reading."
)

falling_type = st.sidebar.radio(
    "RSI Metric for Falling Filter",
    ["ConnorsRSI (CRSI)", "Price RSI (RSI-3)"],
    index=0,
    disabled=not require_falling_rsi
)

# EXIT RULES
st.sidebar.markdown("---")
st.sidebar.subheader("Exit Rules")
x_days = st.sidebar.slider("Exit: Prior X-Day High Lookback", min_value=1, max_value=20, value=5, step=1)
crsi_exit_thresh = st.sidebar.selectbox("Exit: 2-Day CRSI Threshold (> N)", [0, 10, 20, 30, 40, 50, 60, 70, 80, 90], index=7)
exit_mode = st.sidebar.selectbox(
    "Exit Condition Logic",
    ["Either (Whichever occurs first)", "X-Day High Only", "2-Day CRSI Threshold Only", "Both Required"]
)
entry_timing = st.sidebar.radio("Entry Execution", ["Next Open", "Signal Day Close"], horizontal=True)

# ==========================================
# 6. TABS
# ==========================================

tab_universe, tab_single, tab_screener = st.tabs([
    "🌐 20-Year Universe Backtester (Uncapped)",
    "📊 Single Ticker Multi-Bucket & Matrix",
    "🔍 Live Market Screener (Full Universes)"
])

# ------------------------------------------
# TAB 1: 20-YEAR UNIVERSE PORTFOLIO BACKTEST
# ------------------------------------------
with tab_universe:
    st.subheader("20-Year Multi-Asset Portfolio Backtest")
    st.caption(
        f"Simulating trades across selected buckets: **{', '.join(selected_buckets) if selected_buckets else 'None Selected'}** "
        f"| Exit: **{exit_mode}**"
    )
    
    u1, u2, u3 = st.columns([2, 1, 1])
    with u1:
        univ_choice = st.selectbox(
            "Select Universe Basket:",
            [
                "Major Liquid ETFs (Complete List)",
                "Full Nasdaq 100",
                "Full S&P 500",
                "Russell Representative & Growth",
                "Top Global ADRs",
                "Custom Ticker List"
            ]
        )
    with u2:
        start_year = st.selectbox("Start Year", options=list(range(2004, 2026)), index=2)
    with u3:
        initial_cap = st.number_input("Starting Capital ($)", value=25000.0, step=5000.0)

    p1, p2 = st.columns(2)
    with p1:
        max_positions = st.slider("Max Concurrent Open Positions", min_value=1, max_value=30, value=5, step=1)
    with p2:
        custom_univ = st.text_input("Custom Ticker Basket (Comma-separated):", value="SPY, QQQ, DIA, IWM, SMH, XLF, XLE, GLD, TLT")

    if st.button("🚀 Run Universe Backtest", type="primary"):
        if not selected_buckets:
            st.error("Please select at least one ConnorsRSI bucket in the sidebar.")
            st.stop()
            
        # Complete uncapped list loading
        if univ_choice == "Major Liquid ETFs (Complete List)":
            ticker_list = MAJOR_ETFS
        elif univ_choice == "Full Nasdaq 100":
            ticker_list = get_nasdaq100_tickers()
        elif univ_choice == "Full S&P 500":
            ticker_list = get_sp500_tickers()
        elif univ_choice == "Russell Representative & Growth":
            ticker_list = RUSSELL_REPRESENTATIVE
        elif univ_choice == "Top Global ADRs":
            ticker_list = TOP_ADRS
        else:
            ticker_list = [t.strip().upper() for t in custom_univ.split(",") if t.strip()]

        st.info(f"Downloading historical data since {start_year} for all {len(ticker_list)} symbols (NO truncation applied)...")
        
        # Parallel chunked download
        data_dict = {}
        chunk_size = 50
        num_chunks = int(np.ceil(len(ticker_list) / chunk_size))
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        start_dt = datetime(start_year, 1, 1)
        
        for c in range(num_chunks):
            chunk = ticker_list[c * chunk_size : (c + 1) * chunk_size]
            status_text.text(f"Downloading chunk {c + 1}/{num_chunks} ({len(chunk)} tickers)...")
            try:
                b_data = yf.download(
                    chunk,
                    start=start_dt,
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=True,
                    progress=False,
                    threads=True
                )
                for t in chunk:
                    try:
                        if isinstance(b_data.columns, pd.MultiIndex):
                            if t in b_data.columns.get_level_values(0):
                                df_t = b_data[t].dropna(how='all')
                            else:
                                continue
                        else:
                            df_t = b_data.dropna(how='all')
                            
                        if len(df_t) >= 120:
                            data_dict[t] = prepare_crsi_data(df_t, x_days=x_days)
                    except Exception:
                        continue
            except Exception:
                pass
            progress_bar.progress((c + 1) / num_chunks)
            
        progress_bar.empty()
        status_text.empty()
        
        if not data_dict:
            st.error("No valid historical price data retrieved.")
        else:
            with st.spinner("Simulating multi-asset portfolio..."):
                trades_df, equity_df = run_universe_backtest(
                    data_dict=data_dict,
                    selected_buckets=selected_buckets,
                    exit_mode=exit_mode,
                    x_days=x_days,
                    crsi_exit_thresh=crsi_exit_thresh,
                    require_falling=require_falling_rsi,
                    falling_type=falling_type,
                    entry_timing=entry_timing,
                    initial_capital=initial_cap,
                    max_positions=max_positions
                )
                
            if trades_df.empty:
                st.warning("No trades were triggered for the selected bucket(s) during this timeframe.")
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
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=equity_df.index,
                    y=equity_df['Equity'],
                    mode='lines',
                    line=dict(color='#00C805', width=2),
                    name='Portfolio Equity'
                ))
                fig.update_layout(
                    title=f"Portfolio Equity Curve ({start_year} – Present | Buckets: {', '.join(selected_buckets)})",
                    xaxis_title="Date",
                    yaxis_title="Total Equity ($)",
                    template="plotly_white",
                    height=450
                )
                st.plotly_chart(fig, use_container_width=True)
                
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
# TAB 2: SINGLE TICKER MULTI-BUCKET & MATRIX
# ------------------------------------------
with tab_single:
    st.subheader("Single Ticker: Combined Selected Buckets & 10-Bucket Matrix")
    
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        st_ticker = st.text_input("Symbol", value="SPY").upper()
    with s_col2:
        st_start_year = st.selectbox("Historical Start Year", [2004, 2006, 2010, 2015, 2020], index=1, key="st_yr")
        
    if st.button("Run Single Ticker Backtest", type="primary"):
        with st.spinner(f"Evaluating {st_ticker}..."):
            raw_s = yf.download(st_ticker, start=datetime(st_start_year, 1, 1), progress=False, auto_adjust=True)
            if isinstance(raw_s.columns, pd.MultiIndex):
                raw_s.columns = raw_s.columns.get_level_values(0)
                
            if len(raw_s) >= 120:
                df_prep = prepare_crsi_data(raw_s, x_days=x_days)
                falling_col = 'CRSI_Falling' if falling_type == "ConnorsRSI (CRSI)" else 'RSI3_Falling'
                
                # 1. Backtest for the user-selected combined buckets
                comb_mask = get_bucket_mask(df_prep['CRSI_2D'], selected_buckets)
                if require_falling_rsi:
                    comb_mask = comb_mask & df_prep[falling_col]
                    
                comb_trades = []
                in_t = False
                e_price = 0.0
                e_date = None
                cap = 10000.0
                equity_vals = []
                
                for i in range(1, len(df_prep)):
                    bar = df_prep.iloc[i]
                    c_date = df_prep.index[i]
                    
                    if in_t:
                        hit_c = bar['CRSI_2D'] > crsi_exit_thresh
                        hit_xh = pd.notna(bar['X_Day_High']) and (bar['High'] >= bar['X_Day_High'])
                        exit_now = False
                        exit_p = bar['Close']
                        
                        if exit_mode == "Either (Whichever occurs first)" and (hit_c or hit_xh):
                            exit_now = True
                            exit_p = max(bar['Open'], bar['X_Day_High']) if hit_xh else bar['Close']
                        elif exit_mode == "Both Required" and (hit_c and hit_xh):
                            exit_now = True
                        elif exit_mode == "X-Day High Only" and hit_xh:
                            exit_now = True
                            exit_p = max(bar['Open'], bar['X_Day_High'])
                        elif exit_mode == "2-Day CRSI Threshold Only" and hit_c:
                            exit_now = True
                            
                        if exit_now:
                            ret = ((exit_p / e_price) - 1) * 100
                            pnl = (ret / 100) * cap
                            cap += pnl
                            comb_trades.append({
                                'Entry Date': e_date, 'Entry Price': e_price,
                                'Exit Date': c_date, 'Exit Price': exit_p,
                                'Return %': round(ret, 2), 'PnL ($)': round(pnl, 2),
                                'Hold Days': (c_date - e_date).days
                            })
                            in_t = False
                            
                    if not in_t:
                        trig = comb_mask.iloc[i - 1] if entry_timing == "Next Open" else comb_mask.iloc[i]
                        if trig:
                            e_price = bar['Open'] if entry_timing == "Next Open" else bar['Close']
                            e_date = c_date
                            in_t = True
                            
                    equity_vals.append({'Date': c_date, 'Equity': cap})
                    
                # Metrics for combined selected buckets
                st.markdown(f"### Performance for Combined Buckets: `{', '.join(selected_buckets)}`")
                if comb_trades:
                    ct_df = pd.DataFrame(comb_trades)
                    eq_s_df = pd.DataFrame(equity_vals).set_index('Date')
                    
                    n_comb = len(ct_df)
                    wr_comb = (len(ct_df[ct_df['PnL ($)'] > 0]) / n_comb) * 100
                    tot_ret_comb = ((cap / 10000.0) - 1) * 100
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Trades", n_comb)
                    c2.metric("Win Rate", f"{wr_comb:.1f}%")
                    c3.metric("Total Return", f"{tot_ret_comb:.1f}%")
                    c4.metric("Avg Trade Return", f"{ct_df['Return %'].mean():.2f}%")
                    
                    fig_s = go.Figure()
                    fig_s.add_trace(go.Scatter(x=eq_s_df.index, y=eq_s_df['Equity'], mode='lines', line=dict(color='#2962FF', width=2)))
                    fig_s.update_layout(title=f"{st_ticker} Combined Selection Equity Curve", template="plotly_white", height=380)
                    st.plotly_chart(fig_s, use_container_width=True)
                else:
                    st.warning("No trades triggered for the selected combined buckets on this ticker.")
                    
                # 2. Complete 10-Bucket Larry Connors Matrix for Context
                st.markdown("### Larry Connors 10-Bucket Comparative Matrix")
                matrix_rows = []
                for b_name, (lb, hb) in BUCKET_RANGES.items():
                    sig_b = (df_prep['CRSI_2D'] >= lb) & (df_prep['CRSI_2D'] <= hb)
                    if require_falling_rsi:
                        sig_b = sig_b & df_prep[falling_col]
                        
                    t_b = []
                    in_tb = False
                    ep_b = 0.0
                    for j in range(1, len(df_prep)):
                        bj = df_prep.iloc[j]
                        if in_tb:
                            h_c = bj['CRSI_2D'] > crsi_exit_thresh
                            h_xh = pd.notna(bj['X_Day_High']) and (bj['High'] >= bj['X_Day_High'])
                            ex = False
                            exp = bj['Close']
                            if exit_mode == "Either (Whichever occurs first)" and (h_c or h_xh):
                                ex = True
                                exp = max(bj['Open'], bj['X_Day_High']) if h_xh else bj['Close']
                            elif exit_mode == "Both Required" and (h_c and h_xh):
                                ex = True
                            elif exit_mode == "X-Day High Only" and h_xh:
                                ex = True
                                exp = max(bj['Open'], bj['X_Day_High'])
                            elif exit_mode == "2-Day CRSI Threshold Only" and h_c:
                                ex = True
                            if ex:
                                t_b.append(((exp / ep_b) - 1) * 100)
                                in_tb = False
                        if not in_tb:
                            trig_j = sig_b.iloc[j - 1] if entry_timing == "Next Open" else sig_b.iloc[j]
                            if trig_j:
                                ep_b = bj['Open'] if entry_timing == "Next Open" else bj['Close']
                                in_tb = True
                                
                    n_b = len(t_b)
                    wr_b = (sum(1 for r in t_b if r > 0) / n_b * 100) if n_b > 0 else 0.0
                    avg_r_b = np.mean(t_b) if n_b > 0 else 0.0
                    
                    matrix_rows.append({
                        "2-Day CRSI Bucket": b_name,
                        "Selected": "✅ Yes" if b_name in selected_buckets else "",
                        "Total Trades": n_b,
                        "Win Rate %": round(wr_b, 1),
                        "Avg Return %": round(avg_r_b, 2)
                    })
                st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True)

# ------------------------------------------
# TAB 3: LIVE MARKET SCREENER (UNCAPPED)
# ------------------------------------------
with tab_screener:
    st.subheader("Market Screener: Full Universes (Uncapped)")
    st.caption(
        f"Scanning for entry in buckets: **{', '.join(selected_buckets) if selected_buckets else 'None Selected'}** "
        f"| Falling RSI Filter: **{require_falling_rsi}** ({falling_type if require_falling_rsi else 'Off'})"
    )
    
    screen_univs = st.multiselect(
        "Select Universes to Scan Completely:",
        ["Full S&P 500", "Full Nasdaq 100", "Major US ETFs", "Russell Representative", "Top Global ADRs"],
        default=["Full Nasdaq 100", "Major US ETFs"]
    )
    
    only_actionable = st.checkbox("Show Only Buy Signals & Exits", value=False)
    
    if st.button("🚀 Run Live Screener Across All Tickers", type="primary"):
        if not selected_buckets:
            st.error("Please select at least one ConnorsRSI bucket in the sidebar.")
            st.stop()
            
        full_scan_list = []
        if "Full S&P 500" in screen_univs:
            full_scan_list.extend(get_sp500_tickers())
        if "Full Nasdaq 100" in screen_univs:
            full_scan_list.extend(get_nasdaq100_tickers())
        if "Major US ETFs" in screen_univs:
            full_scan_list.extend(MAJOR_ETFS)
        if "Russell Representative" in screen_univs:
            full_scan_list.extend(RUSSELL_REPRESENTATIVE)
        if "Top Global ADRs" in screen_univs:
            full_scan_list.extend(TOP_ADRS)
            
        full_scan_list = sorted(list(set(full_scan_list)))
        st.info(f"Scanning all {len(full_scan_list)} tickers without truncation...")
        
        chunk_size = 50
        num_chunks = int(np.ceil(len(full_scan_list) / chunk_size))
        pbar = st.progress(0.0)
        s_text = st.empty()
        
        results = []
        falling_col = 'CRSI_Falling' if falling_type == "ConnorsRSI (CRSI)" else 'RSI3_Falling'
        
        for c in range(num_chunks):
            chunk = full_scan_list[c * chunk_size : (c + 1) * chunk_size]
            s_text.text(f"Scanning chunk {c + 1}/{num_chunks} ({len(chunk)} tickers)...")
            try:
                b_raw = yf.download(
                    chunk,
                    period="1y",
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=True,
                    progress=False,
                    threads=True
                )
                for t in chunk:
                    try:
                        if isinstance(b_raw.columns, pd.MultiIndex):
                            if t in b_raw.columns.get_level_values(0):
                                df_t = b_raw[t].dropna(how='all')
                            else:
                                continue
                        else:
                            df_t = b_raw.dropna(how='all')
                            
                        if len(df_t) < 120:
                            continue
                            
                        df_calc = prepare_crsi_data(df_t, x_days=x_days)
                        latest = df_calc.iloc[-1]
                        prev = df_calc.iloc[-2]
                        
                        crsi_2d = latest['CRSI_2D']
                        crsi_today = latest['CRSI']
                        rsi3_today = latest['RSI3']
                        is_falling = bool(latest[falling_col])
                        
                        # Multi-bucket check
                        in_selected_bucket = check_value_in_buckets(crsi_2d, selected_buckets)
                        passes_falling = (not require_falling_rsi) or is_falling
                        
                        # Determine current bucket name
                        current_b_name = "None"
                        for b_name, (b_min, b_max) in BUCKET_RANGES.items():
                            if b_min <= crsi_2d <= b_max:
                                current_b_name = b_name
                                break
                                
                        status = "⚪ NEUTRAL"
                        if in_selected_bucket and passes_falling:
                            status = "🟢 BUY SIGNAL"
                        elif crsi_2d > crsi_exit_thresh or (pd.notna(latest['X_Day_High']) and latest['High'] >= latest['X_Day_High']):
                            status = "🚨 EXIT MET"
                            
                        results.append({
                            "Ticker": t,
                            "Status": status,
                            "2-Day CRSI": round(crsi_2d, 2),
                            "Current Bucket": current_b_name,
                            "CRSI Today": round(crsi_today, 2),
                            "CRSI Prev": round(prev['CRSI'], 2),
                            "RSI(3) Today": round(rsi3_today, 2),
                            "Falling?": "✅ Yes" if is_falling else "❌ No",
                            "Close": round(latest['Close'], 2),
                            "X-Day High": round(latest['X_Day_High'], 2) if pd.notna(latest['X_Day_High']) else np.nan
                        })
                    except Exception:
                        continue
            except Exception:
                pass
            pbar.progress((c + 1) / num_chunks)
            
        pbar.empty()
        s_text.empty()
        
        if results:
            res_df = pd.DataFrame(results).sort_values(by="2-Day CRSI")
            if only_actionable:
                res_df = res_df[res_df['Status'] != "⚪ NEUTRAL"]
                
            n_buys = len(res_df[res_df['Status'] == "🟢 BUY SIGNAL"])
            n_exits = len(res_df[res_df['Status'] == "🚨 EXIT MET"])
            
            c_res1, c_res2, c_res3 = st.columns(3)
            c_res1.metric("Total Scanned", len(results))
            c_res2.metric("Buy Signals (In Selected Buckets)", n_buys)
            c_res3.metric("Exit Signals", n_exits)
            
            st.dataframe(res_df, use_container_width=True, height=600)
        else:
            st.warning("No data retrieved.")
