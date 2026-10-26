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
    """Calculates RSI(3), ConnorsRSI(3,2,100), 2-Day CRSI, and X-Day Highs."""
    df = df.copy()
    df['RSI3'] = calculate_rsi(df['Close'], 3)
    rsi_streak = calculate_rsi(calculate_streak(df['Close']), 2)
    pr = calculate_percent_rank(df['Close'], 100)
    
    df['CRSI'] = (df['RSI3'] + rsi_streak + pr) / 3.0
    df['CRSI_2D'] = df['CRSI'].rolling(2).mean()
    df['X_Day_High'] = df['High'].shift(1).rolling(x_days).max()
    
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

def check_value_in_buckets(val: float, selected_buckets: list) -> bool:
    if not selected_buckets or pd.isna(val):
        return False
    for b in selected_buckets:
        lb, hb = BUCKET_RANGES[b]
        if lb <= val <= hb:
            return True
    return False

# ==========================================
# 3. UNIVERSES & LOADERS
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
# 4. UNIVERSE ENGINE WITH COMMISSIONS
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
    commission_per_trade: float
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
                # Deduct commission on exit
                gross_proceeds = pos['shares'] * exit_price
                net_proceeds = gross_proceeds - commission_per_trade
                
                # Net PnL accounts for both entry and exit commissions
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
                    'Commissions ($)': round(total_commissions, 2),
                    'Net Return %': round(net_ret_pct, 2),
                    'Net PnL ($)': round(net_pnl, 2)
                })
                tickers_to_exit.append(ticker)
                
        for t in tickers_to_exit:
            del active_positions[t]
            
        # 2. PROCESS ENTRIES
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
                
                in_bucket = check_value_in_buckets(crsi_2d, selected_buckets)
                passes_falling = True
                if require_falling:
                    passes_falling = bool(bar[falling_col])
                    
                if in_bucket and passes_falling:
                    candidates.append({
                        'Ticker': ticker,
                        'CRSI_2D': crsi_2d
                    })
                    
            if candidates:
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
                        
                    # Reserve cash for purchase and entry commission
                    avail_for_stock = max(min(capital, target_slot_size) - commission_per_trade, 0.0)
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
                            'entry_commission': commission_per_trade
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
# 5. ANNUAL METRICS AGGREGATOR
# ==========================================

def calculate_annual_metrics(equity_df: pd.DataFrame, trades_df: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    """Computes year-by-year portfolio return, win rate, PnL, profit factor, and max drawdown."""
    if equity_df.empty:
        return pd.DataFrame()
        
    equity_df = equity_df.copy()
    equity_df['Year'] = equity_df.index.year
    years = sorted(equity_df['Year'].unique())
    
    if not trades_df.empty:
        trades_df = trades_df.copy()
        trades_
