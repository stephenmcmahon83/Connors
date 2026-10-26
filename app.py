import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import urllib.request

st.set_page_config(
    page_title="ConnorsRSI Pullback Scanner & Backtester",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 1. TECHNICAL INDICATORS (PURE VECTORIZED)
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

def calculate_connors_rsi(close: pd.Series, rsi_len: int = 3, streak_len: int = 2, rank_len: int = 100) -> pd.Series:
    """ConnorsRSI(3, 2, 100)."""
    rsi_price = calculate_rsi(close, rsi_len)
    streak = calculate_streak(close)
    rsi_streak = calculate_rsi(streak, streak_len)
    pct_rank = calculate_percent_rank(close, rank_len)
    return (rsi_price + rsi_streak + pct_rank) / 3.0

def calculate_adx(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """10-day Average Directional Index (ADX) using Wilder's smoothing."""
    high, low, close = df['High'], df['Low'], df['Close']
    prev_close = close.shift(1)
    
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    pos_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    neg_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    
    tr_smooth = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    pos_dm_smooth = pos_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    neg_dm_smooth = neg_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    pos_di = 100 * (pos_dm_smooth / tr_smooth.replace(0, np.nan))
    neg_di = 100 * (neg_dm_smooth / tr_smooth.replace(0, np.nan))
    
    dx = 100 * ((pos_di - neg_di).abs() / (pos_di + neg_di).replace(0, np.nan))
    adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return adx.fillna(0)

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-computes indicators and setup conditions."""
    df = df.copy()
    df['SMA21_Vol'] = df['Volume'].rolling(21).mean()
    df['ADX10'] = calculate_adx(df, 10)
    df['CRSI'] = calculate_connors_rsi(df['Close'], 3, 2, 100)
    
    day_range = (df['High'] - df['Low']).replace(0, np.nan)
    df['Close_Range_Pct'] = ((df['Close'] - df['Low']) / day_range) * 100.0
    df['Intraday_Drop_Pct'] = ((df['Close'].shift(1) - df['Low']) / df['Close'].shift(1)) * 100.0
    return df

# ==========================================
# 2. TICKER UNIVERSE RETRIEVAL
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

RUSSELL_LEADERS = [
    "PLTR", "HOOD", "AFRM", "SOFI", "RBLX", "COIN", "DKNG", "SNAP", "DASH", "PINS",
    "CRWD", "NET", "SNOW", "PANW", "ZS", "MDB", "DDOG", "TTD", "ENPH", "CELH",
    "SYM", "PATH", "IONQ", "SMCI", "ARM", "RIVN", "LCID", "MARA", "RIOT", "CLSK"
]

@st.cache_data(ttl=86400)
def get_sp500_tickers() -> list:
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            tables = pd.read_html(response)
        tickers = tables[0]['Symbol'].str.replace('.', '-', regex=False).tolist()
        return sorted(list(set(tickers)))
    except Exception:
        return ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "BRK-B", "JPM", "V", "UNH"]

@st.cache_data(ttl=86400)
def get_nasdaq100_tickers() -> list:
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            tables = pd.read_html(response)
        tickers = tables[4]['Ticker'].str.replace('.', '-', regex=False).tolist()
        return sorted(list(set(tickers)))
    except Exception:
        return ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "COST", "AVGO", "TSLA", "AMD"]

# ==========================================
# 3. BACKTESTER ENGINE
# ==========================================

def run_backtest(df: pd.DataFrame, W: float, X: float, Y: float, Z: float, N: float, initial_capital: float = 10000.0):
    df = prepare_data(df)
    
    cond1 = df['Close'] > 5.0
    cond2 = df['SMA21_Vol'] >= 250000
    cond3 = df['ADX10'] > 30.0
    cond4 = df['Intraday_Drop_Pct'] >= W
    cond5 = df['Close_Range_Pct'] <= X
    cond6 = df['CRSI'] < Y
    
    df['Setup_Signal'] = cond1 & cond2 & cond3 & cond4 & cond5 & cond6
    
    trades = []
    in_trade = False
    entry_price = 0.0
    entry_date = None
    shares = 0
    capital = initial_capital
    portfolio_values = []
    
    dates = df.index
    for i in range(1, len(df)):
        curr_date = dates[i]
        
        # Rule 8: Exit at close when CRSI > N
        if in_trade:
            if df['CRSI'].iloc[i] > N:
                exit_price = df['Close'].iloc[i]
                pnl = (exit_price - entry_price) * shares
                ret_pct = ((exit_price / entry_price) - 1) * 100
                capital += (shares * exit_price)
                
                trades.append({
                    'Entry Date': entry_date,
                    'Entry Price': round(entry_price, 2),
                    'Exit Date': curr_date,
                    'Exit Price': round(exit_price, 2),
                    'Hold Days': (curr_date - entry_date).days,
                    'Return %': round(ret_pct, 2),
                    'PnL ($)': round(pnl, 2),
                    'Equity': round(capital, 2)
                })
                in_trade = False
                shares = 0
        
        # Rule 7: Entry limit order Z% below yesterday's close
        if not in_trade and df['Setup_Signal'].iloc[i - 1]:
            limit_price = df['Close'].iloc[i - 1] * (1 - (Z / 100.0))
            if df['Low'].iloc[i] <= limit_price:
                entry_price = min(df['Open'].iloc[i], limit_price)
                entry_date = curr_date
                shares = int(capital // entry_price)
                if shares > 0:
                    capital -= (shares * entry_price)
                    in_trade = True
        
        current_equity = capital + (shares * df['Close'].iloc[i] if in_trade else 0)
        portfolio_values.append({'Date': curr_date, 'Equity': current_equity})
        
    equity_df = pd.DataFrame(portfolio_values).set_index('Date') if portfolio_values else pd.DataFrame()
    trades_df = pd.DataFrame(trades)
    return trades_df, equity_df

# ==========================================
# 4. SIDEBAR CONFIGURATION (UPDATED RANGES)
# ==========================================

st.sidebar.header("Strategy Rules Parameters")

# W: 0 to 10, < 10
W = st.sidebar.slider(
    "Rule 4: Low % Drop Below Prev Close (W)",
    min_value=0.0,
    max_value=9.9,
    value=4.0,
    step=0.1,
    help="Today's low must drop at least W% below yesterday's close (< 10%)."
)

# X: 0 to 100, step 5
X = st.sidebar.slider(
    "Rule 5: Max % of Day's Range for Close (X)",
    min_value=0,
    max_value=100,
    value=25,
    step=5,
    help="Today's close must sit in the bottom X% of the day's total range."
)

# Y: 5 to 95, step 5
Y = st.sidebar.slider(
    "Rule 6: ConnorsRSI Entry Threshold (< Y)",
    min_value=5,
    max_value=95,
    value=10,
    step=5,
    help="ConnorsRSI(3,2,100) must close below Y."
)

# Z: 0 to 10
Z = st.sidebar.slider(
    "Rule 7: Tomorrow's Limit % Below Close (Z)",
    min_value=0.0,
    max_value=10.0,
    value=4.0,
    step=0.5,
    help="Intraday limit order placed Z% below today's closing price."
)

# N: 0 to 90, step 10
N = st.sidebar.slider(
    "Rule 8: ConnorsRSI Exit Threshold (> N)",
    min_value=0,
    max_value=90,
    value=70,
    step=10,
    help="Exit position at market close when ConnorsRSI crosses above N."
)

# ==========================================
# 5. TABS: SCANNER & BACKTESTER
# ==========================================

tab1, tab2 = st.tabs(["🔍 Market Scanner (Tomorrow's Setups)", "📊 Strategy Backtester"])

# ------------------------------------------
# TAB 1: SCANNER
# ------------------------------------------
with tab1:
    st.subheader("ConnorsRSI Pullback Scanner")
    st.markdown("Scans selected indices and universes for stocks meeting Rules 1–6 on the latest market close.")
    
    selected_universes = st.multiselect(
        "Select Universes to Include in Scan:",
        ["S&P 500", "Nasdaq 100", "Russell Leaders / Growth", "Major US ETFs", "Top Global ADRs"],
        default=["S&P 500", "Nasdaq 100", "Major US ETFs"]
    )
    
    custom_tickers = st.text_input("Add Extra Custom Tickers (Comma-separated, optional):", value="")
    
    col_scan, col_cap = st.columns([1, 2])
    with col_scan:
        start_scan = st.button("🚀 Run Market Scan", type="primary")
    with col_cap:
        scan_limit = st.slider("Max Tickers to Scan (prevents API rate limits)", min_value=50, max_value=800, value=300, step=50)

    if start_scan:
        universe_list = []
        if "S&P 500" in selected_universes:
            universe_list.extend(get_sp500_tickers())
        if "Nasdaq 100" in selected_universes:
            universe_list.extend(get_nasdaq100_tickers())
        if "Russell Leaders / Growth" in selected_universes:
            universe_list.extend(RUSSELL_LEADERS)
        if "Major US ETFs" in selected_universes:
            universe_list.extend(MAJOR_ETFS)
        if "Top Global ADRs" in selected_universes:
            universe_list.extend(TOP_ADRS)
            
        if custom_tickers.strip():
            universe_list.extend([t.strip().upper() for t in custom_tickers.split(",") if t.strip()])
            
        # Deduplicate and apply limit
        all_tickers = sorted(list(set(universe_list)))[:scan_limit]
        
        st.info(f"Scanning {len(all_tickers)} tickers in fast parallel batches...")
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        candidates = []
        batch_size = 50
        num_batches = int(np.ceil(len(all_tickers) / batch_size))
        
        for b in range(num_batches):
            batch = all_tickers[b * batch_size : (b + 1) * batch_size]
            status_text.text(f"Fetching batch {b + 1}/{num_batches} ({len(batch)} tickers)...")
            
            try:
                # Batched download for maximum performance
                batch_data = yf.download(
                    tickers=batch,
                    period="1y",
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=True,
                    progress=False,
                    threads=True
                )
                
                for ticker in batch:
                    try:
                        # Extract ticker data from single or multi-index structure
                        if isinstance(batch_data.columns, pd.MultiIndex):
                            if ticker in batch_data.columns.get_level_values(0):
                                df_t = batch_data[ticker].dropna(how='all')
                            elif ticker in batch_data.columns.get_level_values(1):
                                df_t = batch_data.xs(ticker, axis=1, level=1).dropna(how='all')
                            else:
                                continue
                        else:
                            df_t = batch_data.dropna(how='all')
                            
                        if len(df_t) >= 120:
                            df_calc = prepare_data(df_t)
                            latest = df_calc.iloc[-1]
                            
                            # Rule 1 to 6 Validation
                            r1 = latest['Close'] > 5.0
                            r2 = latest['SMA21_Vol'] >= 250000
                            r3 = latest['ADX10'] > 30.0
                            r4 = latest['Intraday_Drop_Pct'] >= W
                            r5 = latest['Close_Range_Pct'] <= X
                            r6 = latest['CRSI'] < Y
                            
                            if r1 and r2 and r3 and r4 and r5 and r6:
                                limit_buy = latest['Close'] * (1 - (Z / 100.0))
                                candidates.append({
                                    "Ticker": ticker,
                                    "Close Price": f"${latest['Close']:.2f}",
                                    "Tomorrow Limit Buy": f"${limit_buy:.2f}",
                                    "ConnorsRSI": f"{latest['CRSI']:.2f}",
                                    "ADX(10)": f"{latest['ADX10']:.2f}",
                                    "Range %": f"{latest['Close_Range_Pct']:.1f}%",
                                    "Drop from Prev Close %": f"{latest['Intraday_Drop_Pct']:.1f}%",
                                    "21-Day Avg Vol": f"{int(latest['SMA21_Vol']):,}"
                                })
                    except Exception:
                        continue
            except Exception:
                pass
            
            progress_bar.progress((b + 1) / num_batches)
            
        progress_bar.empty()
        status_text.empty()
        
        if candidates:
            st.success(f"Found {len(candidates)} candidate(s) triggering entry orders for tomorrow!")
            st.dataframe(pd.DataFrame(candidates), use_container_width=True)
        else:
            st.warning("No tickers met all 6 entry criteria under the current parameter thresholds.")

# ------------------------------------------
# TAB 2: BACKTESTER
# ------------------------------------------
with tab2:
    st.subheader("Historical Strategy Backtester")
    
    b_col1, b_col2, b_col3 = st.columns(3)
    with b_col1:
        bt_ticker = st.text_input("Ticker Symbol", value="TQQQ").upper()
    with b_col2:
        start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=5*365))
    with b_col3:
        initial_capital = st.number_input("Starting Capital ($)", value=10000.0, step=1000.0)
        
    if st.button("📈 Run Historical Backtest", type="primary"):
        with st.spinner(f"Testing {bt_ticker}..."):
            data = yf.download(bt_ticker, start=start_date, interval="1d", progress=False, auto_adjust=True)
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
                
            if len(data) < 120:
                st.error("Insufficient historical price history. Select an earlier start date.")
            else:
                trades_df, equity_df = run_backtest(
                    data, W=W, X=X, Y=Y, Z=Z, N=N, initial_capital=initial_capital
                )
                
                if trades_df.empty:
                    st.warning("No trades triggered under these parameter values.")
                else:
                    total_trades = len(trades_df)
                    wins = trades_df[trades_df['PnL ($)'] > 0]
                    win_rate = (len(wins) / total_trades) * 100.0
                    total_return = ((equity_df['Equity'].iloc[-1] / initial_capital) - 1) * 100.0
                    
                    gross_profit = trades_df.loc[trades_df['PnL ($)'] > 0, 'PnL ($)'].sum()
                    gross_loss = abs(trades_df.loc[trades_df['PnL ($)'] < 0, 'PnL ($)'].sum())
                    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.nan
                    
                    rolling_max = equity_df['Equity'].cummax()
                    drawdown = (equity_df['Equity'] - rolling_max) / rolling_max
                    max_dd = drawdown.min() * 100.0
                    
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Total Trades", total_trades)
                    m2.metric("Win Rate", f"{win_rate:.1f}%")
                    m3.metric("Total Return", f"{total_return:.1f}%")
                    m4.metric("Profit Factor", f"{profit_factor:.2f}" if not np.isnan(profit_factor) else "N/A")
                    m5.metric("Max Drawdown", f"{max_dd:.1f}%")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=equity_df.index,
                        y=equity_df['Equity'],
                        mode='lines',
                        name='Portfolio Value',
                        line=dict(color='#00C805', width=2)
                    ))
                    fig.update_layout(
                        title=f"{bt_ticker} Strategy Equity Curve",
                        xaxis_title="Date",
                        yaxis_title="Portfolio Equity ($)",
                        template="plotly_white",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("Executed Trade Log")
                    st.dataframe(trades_df, use_container_width=True)
