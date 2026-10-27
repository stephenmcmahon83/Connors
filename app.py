import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

st.set_page_config(
    page_title="ConnorsRSI Universe Backtester, Screener & Charts",
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

def calculate_atr_pct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculates Wilder's 14-day ATR expressed as a percentage of the closing price."""
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    atr_pct = (atr / close.replace(0, np.nan)) * 100.0
    return atr_pct.fillna(0.0)

def prepare_crsi_data(df: pd.DataFrame, x_days: int) -> pd.DataFrame:
    """Calculates RSI(3), ConnorsRSI(3,2,100), 2-Day CRSI, ATR(14)%, and X-Day Highs."""
    df = df.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
        
    df['RSI3'] = calculate_rsi(df['Close'], 3)
    rsi_streak = calculate_rsi(calculate_streak(df['Close']), 2)
    pr = calculate_percent_rank(df['Close'], 100)
    
    df['CRSI'] = (df['RSI3'] + rsi_streak + pr) / 3.0
    df['CRSI_2D'] = df['CRSI'].rolling(2).mean()
    df['ATR_Pct_14'] = calculate_atr_pct(df, 14)
    df['X_Day_High'] = df['High'].shift(1).rolling(x_days).max()
    
    df['CRSI_Falling'] = df['CRSI'] < df['CRSI'].shift(1)
    df['RSI3_Falling'] = df['RSI3'] < df['RSI3'].shift(1)
    return df

# ==========================================
# 2. VIX REGIME FILTER ENGINE
# ==========================================

@st.cache_data(ttl=3600)
def load_vix_data(start_date: datetime) -> pd.DataFrame:
    try:
        vix = yf.download("^VIX", start=start_date - timedelta(days=60), progress=False, auto_adjust=True)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        if vix.index.tz is not None:
            vix.index = vix.index.tz_localize(None)
        return vix
    except Exception:
        return pd.DataFrame()

def compute_vix_blocked_dates(vix_df: pd.DataFrame, mode: str, lookback: int = 20, crisis_cap: float = 40.0) -> set:
    if vix_df.empty:
        return set()
        
    vix = vix_df.copy()
    vix_close = vix['Close']
    vix_high_n = vix_close.rolling(lookback).max()
    vix_sma_n = vix_close.rolling(lookback).mean()
    
    if mode == "Crisis Cap Only (VIX > Cap)":
        blocked_mask = vix_close >= crisis_cap
    elif mode == "New 20-Day High Close":
        blocked_mask = vix_close >= vix_high_n
    elif mode == "Dual: New 20d High OR Crisis Cap":
        blocked_mask = (vix_close >= vix_high_n) | (vix_close >= crisis_cap)
    elif mode == "VIX Stretch / Spike (VIX > 1.15x 20d SMA)":
        blocked_mask = (vix_close / vix_sma_n) >= 1.15
    else:
        blocked_mask = pd.Series(False, index=vix.index)
        
    return set(vix[blocked_mask].index)

# ==========================================
# 3. PLOTTING: CANDLESTICK WITH TRADE TICKS & 2-DAY CRSI
# ==========================================

def create_candlestick_crsi_chart(df: pd.DataFrame, ticker: str, x_days: int, trades_df: pd.DataFrame = None) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.70, 0.30],
        subplot_titles=(f"{ticker} Candlestick Chart (System Entries & Exits)", "2-Day Average ConnorsRSI (CRSI_2D)")
    )
    
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Price",
            increasing_line_color="#00C805",
            decreasing_line_color="#FF333A"
        ),
        row=1, col=1
    )
    
    if 'X_Day_High' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['X_Day_High'],
                mode='lines',
                line=dict(color='rgba(255, 152, 0, 0.75)', dash='dash', width=1.3),
                name=f"{x_days}-Day High Exit"
            ),
            row=1, col=1
        )
        
    if trades_df is not None and not trades_df.empty:
        t_subset = trades_df[trades_df['Ticker'] == ticker] if 'Ticker' in trades_df.columns else trades_df
            
        win_entries_x, win_entries_y, win_entries_text = [], [], []
        win_exits_x, win_exits_y, win_exits_text = [], [], []
        loss_entries_x, loss_entries_y, loss_entries_text = [], [], []
        loss_exits_x, loss_exits_y, loss_exits_text = [], [], []
        win_lines_x, win_lines_y = [], []
        loss_lines_x, loss_lines_y = [], []
        
        df_index_set = set(df.index)
        
        for _, trade in t_subset.iterrows():
            e_date = trade['Entry Date']
            x_date = trade['Exit Date']
            e_price = trade['Entry Price']
            x_price = trade['Exit Price']
            
            pnl = trade.get('Net PnL ($)', trade.get('PnL ($)', 0.0))
            ret = trade.get('Net Return %', trade.get('Return %', 0.0))
            reason = trade.get('Exit Reason', 'Exit Trigger')
            is_win = pnl > 0
            
            if e_date in df_index_set:
                txt_e = (
                    f"<b>🟢 BUY ENTRY</b><br>"
                    f"Date: {pd.to_datetime(e_date).strftime('%Y-%m-%d')}<br>"
                    f"Price: ${e_price:.2f}<br>"
                    f"Result: {'WIN (+)' if is_win else 'LOSS (-)'} ({ret:+.2f}%)"
                )
                if is_win:
                    win_entries_x.append(e_date)
                    win_entries_y.append(e_price)
                    win_entries_text.append(txt_e)
                else:
                    loss_entries_x.append(e_date)
                    loss_entries_y.append(e_price)
                    loss_entries_text.append(txt_e)
                    
            if x_date in df_index_set:
                txt_x = (
                    f"<b>🔴 SELL EXIT</b><br>"
                    f"Date: {pd.to_datetime(x_date).strftime('%Y-%m-%d')}<br>"
                    f"Price: ${x_price:.2f}<br>"
                    f"Net PnL: ${pnl:+.2f} ({ret:+.2f}%)<br>"
                    f"Reason: {reason}"
                )
                if is_win:
                    win_exits_x.append(x_date)
                    win_exits_y.append(x_price)
                    win_exits_text.append(txt_x)
                else:
                    loss_exits_x.append(x_date)
                    loss_exits_y.append(x_price)
                    loss_exits_text.append(txt_x)
                    
            if e_date in df_index_set and x_date in df_index_set:
                if is_win:
                    win_lines_x.extend([e_date, x_date, None])
                    win_lines_y.extend([e_price, x_price, None])
                else:
                    loss_lines_x.extend([e_date, x_date, None])
                    loss_lines_y.extend([e_price, x_price, None])
                    
        if win_lines_x:
            fig.add_trace(go.Scatter(
                x=win_lines_x, y=win_lines_y, mode='lines',
                line=dict(color='rgba(0, 200, 5, 0.45)', width=1.5, dash='dot'),
                name="Win Path", hoverinfo='skip'
            ), row=1, col=1)
            
        if loss_lines_x:
            fig.add_trace(go.Scatter(
                x=loss_lines_x, y=loss_lines_y, mode='lines',
                line=dict(color='rgba(255, 51, 58, 0.45)', width=1.5, dash='dot'),
                name="Loss Path", hoverinfo='skip'
            ), row=1, col=1)
            
        if win_entries_x:
            fig.add_trace(go.Scatter(
                x=win_entries_x, y=win_entries_y, mode='markers',
                marker=dict(symbol='triangle-up', size=11, color='#00C805', line=dict(width=1, color='#004D1A')),
                name="Win Entry", hovertext=win_entries_text, hoverinfo='text'
            ), row=1, col=1)
            
        if win_exits_x:
            fig.add_trace(go.Scatter(
                x=win_exits_x, y=win_exits_y, mode='markers',
                marker=dict(symbol='triangle-down', size=11, color='#00C805', line=dict(width=1, color='#004D1A')),
                name="Win Exit", hovertext=win_exits_text, hoverinfo='text'
            ), row=1, col=1)
            
        if loss_entries_x:
            fig.add_trace(go.Scatter(
                x=loss_entries_x, y=loss_entries_y, mode='markers',
                marker=dict(symbol='triangle-up', size=11, color='#FF333A', line=dict(width=1, color='#660000')),
                name="Loss Entry", hovertext=loss_entries_text, hoverinfo='text'
            ), row=1, col=1)
            
        if loss_exits_x:
            fig.add_trace(go.Scatter(
                x=loss_exits_x, y=loss_exits_y, mode='markers',
                marker=dict(symbol='triangle-down', size=11, color='#FF333A', line=dict(width=1, color='#660000')),
                name="Loss Exit", hovertext=loss_exits_text, hoverinfo='text'
            ), row=1, col=1)
            
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['CRSI_2D'],
            mode='lines',
            line=dict(color='#2962FF', width=2),
            name="2-Day CRSI"
        ),
        row=2, col=1
    )
    
    crsi_levels = [
        (10, "rgba(0, 200, 5, 0.9)", "dash"),
        (20, "rgba(0, 200, 5, 0.6)", "dot"),
        (30, "rgba(0, 200, 5, 0.4)", "dot"),
        (70, "rgba(255, 59, 48, 0.4)", "dot"),
        (80, "rgba(255, 59, 48, 0.6)", "dot"),
        (90, "rgba(255, 59, 48, 0.9)", "dash"),
    ]
    
    for level, color, dash in crsi_levels:
        fig.add_hline(
            y=level,
            line_color=color,
            line_dash=dash,
            line_width=1.3,
            annotation_text=str(level),
            annotation_position="top right",
            annotation_font_size=10,
            annotation_font_color=color,
            row=2, col=1
        )
        
    fig.update_layout(
        height=720,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(t=50, b=30, l=40, r=40),
        xaxis1=dict(rangeslider=dict(visible=False)),
        yaxis1=dict(title="Price ($)"),
        yaxis2=dict(
            title="CRSI (2-Day Avg)",
            range=[-2, 102],
            tickvals=[0, 10, 20, 30, 50, 70, 80, 90, 100]
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# ==========================================
# 4. BUCKET UTILITIES
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

def check_value_in_buckets(val: float, selected_buckets: list) -> bool:
    if not selected_buckets or pd.isna(val):
        return False
    for b in selected_buckets:
        lb, hb = BUCKET_RANGES[b]
        if lb <= val <= hb:
            return True
    return False

# ==========================================
# 5. UNIVERSES & LOADERS
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
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df_sp = pd.read_csv(url)
        return sorted(df_sp['Symbol'].str.replace('.', '-', regex=False).tolist())
    except Exception:
        return ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "BRK-B", "JPM", "V", "UNH"]

@st.cache_data(ttl=86400)
def get_nasdaq100_tickers() -> list:
    try:
        url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq/nasdaq_tickers.txt"
        df_ndx = pd.read_csv(url, header=None, names=['Symbol'])
        return sorted(df_ndx['Symbol'].str.replace('.', '-', regex=False).tolist()[:100])
    except Exception:
        return ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "COST", "AVGO", "TSLA", "AMD"]

# ==========================================
# 6. UNIVERSE ENGINE WITH RANKING SELECTOR
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
    max_positions: int,
    commission_per_trade: float,
    vix_blocked_dates: set = None,
    max_bars_stop: int = 0,
    rank_method: str = "Lowest 2-Day ConnorsRSI"
):
    all_dates = sorted(list(set().union(*[df.index.tolist() for df in data_dict.values()])))
    if len(all_dates) < 120:
        return pd.DataFrame(), pd.DataFrame()

    capital = initial_capital
    active_positions = {}
    trades = []
    portfolio_equity = []
    vix_blocked = vix_blocked_dates if vix_blocked_dates is not None else set()
    
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
            c_close = curr_bar.get('Close', np.nan)
            c_open = curr_bar.get('Open', np.nan)
            c_high = curr_bar.get('High', np.nan)
            c_crsi_2d = curr_bar.get('CRSI_2D', np.nan)
            x_high = curr_bar.get('X_Day_High', np.nan)
            
            if pd.isna(c_close) or pd.isna(c_open) or pd.isna(c_crsi_2d):
                continue
            
            pos['bars_held'] = pos.get('bars_held', 0) + 1
            
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
                    exit_reason = f"{x_days}-Day High"
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
                exit_reason = f"{x_days}-Day High"
                should_exit = True
            elif exit_mode == "2-Day CRSI Threshold Only" and hit_crsi:
                exit_price = c_close
                exit_reason = f"CRSI 2D > {crsi_exit_thresh}"
                should_exit = True
                
            if not should_exit and (max_bars_stop > 0) and (pos['bars_held'] >= max_bars_stop):
                should_exit = True
                exit_price = c_close
                exit_reason = f"Time Stop ({max_bars_stop} Bars)"
                
            if should_exit and pd.notna(exit_price) and exit_price > 0:
                gross_proceeds = pos['shares'] * exit_price
                net_proceeds = gross_proceeds - commission_per_trade
                total_commissions = pos['entry_commission'] + commission_per_trade
                net_pnl = gross_proceeds - pos['gross_cost'] - total_commissions
                net_ret_pct = (net_pnl / pos['gross_cost']) * 100.0
                
                capital += net_proceeds
                
                trades.append({
                    'Ticker': ticker,
                    'Entry Date': pos['entry_date'],
                    'Entry Price': round(pos['entry_price'], 2),
                    'Exit Date': curr_date,
                    'Exit Price': round(exit_price, 2),
                    'Exit Reason': exit_reason,
                    'Shares': pos['shares'],
                    'Hold Days': (curr_date - pos['entry_date']).days,
                    'Hold Bars': pos['bars_held'],
                    'Setup 2D-CRSI': round(pos.get('setup_crsi', 0.0), 2),
                    'Setup ATR(14)%': round(pos.get('setup_atr_pct', 0.0), 2),
                    'Commissions ($)': round(total_commissions, 2),
                    'Net Return %': round(net_ret_pct, 2),
                    'Net PnL ($)': round(net_pnl, 2)
                })
                tickers_to_exit.append(ticker)
                
        for t in tickers_to_exit:
            del active_positions[t]
            
        # 2. PROCESS ENTRIES (WITH SAME-DAY OVERLAP GUARD & RANKING FILTER)
        available_slots = max_positions - len(active_positions)
        setup_eval_date = prev_date if entry_timing == "Next Open" else curr_date
        vix_is_blocking = setup_eval_date in vix_blocked
        
        if available_slots > 0 and selected_buckets and not vix_is_blocking:
            candidates = []
            
            for ticker, df in data_dict.items():
                # Prevent same-day re-entry if already held or just exited today
                if ticker in active_positions or ticker in tickers_to_exit:
                    continue
                    
                if setup_eval_date not in df.index:
                    continue
                    
                bar = df.loc[setup_eval_date]
                crsi_2d = bar.get('CRSI_2D', np.nan)
                atr_pct = bar.get('ATR_Pct_14', np.nan)
                
                if pd.isna(crsi_2d) or pd.isna(atr_pct):
                    continue
                
                in_bucket = check_value_in_buckets(crsi_2d, selected_buckets)
                passes_falling = True
                if require_falling:
                    passes_falling = bool(bar.get(falling_col, False))
                    
                if in_bucket and passes_falling:
                    candidates.append({
                        'Ticker': ticker,
                        'CRSI_2D': crsi_2d,
                        'ATR_Pct_14': atr_pct
                    })
                    
            if candidates:
                # Apply user-selected sorting method
                if rank_method == "Highest 14-Day ATR (%)":
                    candidates = sorted(candidates, key=lambda c: c['ATR_Pct_14'], reverse=True)[:available_slots]
                else:
                    candidates = sorted(candidates, key=lambda c: c['CRSI_2D'], reverse=False)[:available_slots]
                
                current_unrealized = 0.0
                for t, pos in active_positions.items():
                    if curr_date in data_dict[t].index:
                        c_val = data_dict[t].loc[curr_date].get('Close', np.nan)
                        current_unrealized += pos['shares'] * (c_val if pd.notna(c_val) and c_val > 0 else pos['entry_price'])
                    else:
                        current_unrealized += pos['gross_cost']
                        
                total_equity = max(capital + current_unrealized, 0.0)
                target_slot_size = total_equity / max_positions
                
                for cand in candidates:
                    t = cand['Ticker']
                    df = data_dict[t]
                    if curr_date not in df.index:
                        continue
                    
                    target_col = 'Open' if entry_timing == "Next Open" else 'Close'
                    buy_price = df.loc[curr_date].get(target_col, np.nan)
                    
                    if pd.isna(buy_price) or buy_price <= 0:
                        continue
                        
                    avail_for_stock = min(capital, target_slot_size) - commission_per_trade
                    if pd.isna(avail_for_stock) or avail_for_stock <= 0:
                        continue
                        
                    shares = int(avail_for_stock // buy_price)
                    
                    if shares > 0:
                        gross_cost = shares * buy_price
                        total_outlay = gross_cost + commission_per_trade
                        capital -= total_outlay
                        
                        active_positions[t] = {
                            'entry_price': buy_price,
                            'entry_date': curr_date,
                            'shares': shares,
                            'gross_cost': gross_cost,
                            'entry_commission': commission_per_trade,
                            'bars_held': 0,
                            'setup_crsi': cand['CRSI_2D'],
                            'setup_atr_pct': cand['ATR_Pct_14']
                        }
        
        # 3. DAILY PORTFOLIO TRACKING
        open_val = 0.0
        for t, pos in active_positions.items():
            if curr_date in data_dict[t].index:
                c_p = data_dict[t].loc[curr_date].get('Close', np.nan)
                open_val += pos['shares'] * (c_p if pd.notna(c_p) and c_p > 0 else pos['entry_price'])
            else:
                open_val += pos['gross_cost']
                
        portfolio_equity.append({'Date': curr_date, 'Equity': capital + open_val})
        
    equity_df = pd.DataFrame(portfolio_equity).set_index('Date') if portfolio_equity else pd.DataFrame()
    trades_df = pd.DataFrame(trades)
    return trades_df, equity_df

# ==========================================
# 7. ANNUAL METRICS AGGREGATOR
# ==========================================

def calculate_annual_metrics(equity_df: pd.DataFrame, trades_df: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    if equity_df.empty:
        return pd.DataFrame()
        
    equity_df = equity_df.copy()
    equity_df['Year'] = equity_df.index.year
    years = sorted(equity_df['Year'].unique())
    
    if not trades_df.empty:
        trades_df = trades_df.copy()
        trades_df['Exit Year'] = pd.to_datetime(trades_df['Exit Date']).dt.year
    else:
        trades_df = pd.DataFrame(columns=['Exit Year', 'Net PnL ($)'])
        
    annual_data = []
    
    for idx, yr in enumerate(years):
        yr_equity = equity_df[equity_df['Year'] == yr]['Equity']
        
        if idx == 0:
            start_eq = initial_capital
        else:
            prev_yr = years[idx - 1]
            prev_yr_equity = equity_df[equity_df['Year'] == prev_yr]['Equity']
            start_eq = prev_yr_equity.iloc[-1]
            
        end_eq = yr_equity.iloc[-1]
        yr_return = ((end_eq / start_eq) - 1) * 100.0
        
        roll_max = yr_equity.cummax()
        dd = (yr_equity - roll_max) / roll_max
        max_dd = dd.min() * 100.0
        
        yr_trades = trades_df[trades_df['Exit Year'] == yr]
        n_trades = len(yr_trades)
        
        if n_trades > 0:
            wins = yr_trades[yr_trades['Net PnL ($)'] > 0]
            losses = yr_trades[yr_trades['Net PnL ($)'] < 0]
            win_rate = (len(wins) / n_trades) * 100.0
            net_pnl = yr_trades['Net PnL ($)'].sum()
            
            gross_p = wins['Net PnL ($)'].sum()
            gross_l = abs(losses['Net PnL ($)'].sum())
            profit_factor = (gross_p / gross_l) if gross_l > 0 else (np.nan if gross_p == 0 else 999.0)
        else:
            win_rate = 0.0
            net_pnl = 0.0
            profit_factor = np.nan
            
        annual_data.append({
            'Year': int(yr),
            'Return %': round(yr_return, 2),
            'Net PnL ($)': round(net_pnl, 2),
            'Trades': n_trades,
            'Win Rate %': round(win_rate, 1) if n_trades > 0 else "-",
            'Profit Factor': round(profit_factor, 2) if pd.notna(profit_factor) and profit_factor != 999.0 else ("∞" if profit_factor == 999.0 else "-"),
            'Max Drawdown %': round(max_dd, 1),
            'Ending Equity ($)': round(end_eq, 2)
        })
        
    return pd.DataFrame(annual_data)

# ==========================================
# 8. SIDEBAR CONTROLS
# ==========================================

st.sidebar.title("Strategy Rules")

selected_buckets = st.sidebar.multiselect(
    "1. Entry 2-Day ConnorsRSI Buckets",
    options=list(BUCKET_RANGES.keys()),
    default=["0 - 10", "11 - 20"],
    help="Positions trigger when the 2-day CRSI closes in any of these buckets."
)

if not selected_buckets:
    st.sidebar.warning("Select at least one bucket to generate entry signals.")

# NEW: RANKING / TIE-BREAKER SELECTOR
st.sidebar.markdown("---")
st.sidebar.subheader("🏆 Candidate Ranking / Tie-Breaker")
rank_method = st.sidebar.radio(
    "Sort Signals By (When Signals > Slots):",
    options=[
        "Lowest 2-Day ConnorsRSI",
        "Highest 14-Day ATR (%)"
    ],
    index=0,
    help="Determines which stocks get priority when more setups trigger than available portfolio slots."
)

# ---------------- V
