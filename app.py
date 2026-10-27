import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import urllib.request

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
# 2. PLOTTING: CANDLESTICK & 2-DAY CRSI SUBPLOT
# ==========================================

def create_candlestick_crsi_chart(df: pd.DataFrame, ticker: str, x_days: int) -> go.Figure:
    """
    Creates a synchronized 2-row subplot:
    Row 1: Candlestick chart + Prior X-Day High line
    Row 2: 2-Day Average ConnorsRSI with horizontal thresholds at 10, 20, 30, 70, 80, 90
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.68, 0.32],
        subplot_titles=(f"{ticker} Candlestick Price Chart", "2-Day Average ConnorsRSI (CRSI_2D)")
    )
    
    # 1. Candlestick Price Trace
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
    
    # Prior X-Day High Reference Line
    if 'X_Day_High' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['X_Day_High'],
                mode='lines',
                line=dict(color='rgba(255, 152, 0, 0.8)', dash='dash', width=1.5),
                name=f"{x_days}-Day High Exit"
            ),
            row=1, col=1
        )
        
    # 2. 2-Day CRSI Line
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
    
    # Horizontal Threshold Lines on Row 2 (10, 20, 30, 70, 80, 90)
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
        height=680,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(t=50, b=30, l=40, r=40),
        xaxis1=dict(rangeslider=dict(visible=False)),
        yaxis1=dict(title="Stock Price ($)"),
        yaxis2=dict(
            title="CRSI (2-Day Avg)",
            range=[-2, 102],
            tickvals=[0, 10, 20, 30, 50, 70, 80, 90, 100]
        )
    )
    return fig

# ==========================================
# 3. BUCKET UTILITIES
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
    if not selected_buckets:
        return pd.Series(False, index=series.index)
    mask = pd.Series(False, index=series.index)
    for b in selected_buckets:
        lb, hb = BUCKET_RANGES[b]
        mask = mask | ((series >= lb) & (series <= hb))
    return mask

def check_value_in_buckets(val: float, selected_buckets: list) -> bool:
    if not selected_buckets or pd.isna(val):
        return False
    for b in selected_buckets:
        lb, hb = BUCKET_RANGES[b]
        if lb <= val <= hb:
            return True
    return False

# ==========================================
# 4. UNIVERSES & LOADERS
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
