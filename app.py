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
# 2. PLOTTING: CANDLESTICK WITH TRADE TICKS & 2-DAY CRSI
# ==========================================

def create_candlestick_crsi_chart(df: pd.DataFrame, ticker: str, x_days: int, trades_df: pd.DataFrame = None) -> go.Figure:
    """
    Creates a synchronized 2-row subplot:
    Row 1: Candlestick chart + Prior X-Day High + Color-Coded Entry & Exit Tick Marks
    Row 2: 2-Day Average ConnorsRSI with horizontal thresholds at 10, 20, 30, 70, 80, 90
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.70, 0.30],
        subplot_titles=(f"{ticker} Candlestick Chart (System Entries & Exits)", "2-Day Average ConnorsRSI (CRSI_2D)")
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
                line=dict(color='rgba(255, 152, 0, 0.75)', dash='dash', width=1.3),
                name=f"{x_days}-Day High Exit"
            ),
            row=1, col=1
        )
        
    # 2. System Trade Entries & Exits
    if trades_df is not None and not trades_df.empty:
        if 'Ticker' in trades_df.columns:
            t_subset = trades_df[trades_df['Ticker'] == ticker]
        else:
            t_subset = trades_df
            
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
            
            # Entry tick mark
            if e_date in df_index_set:
                txt_e = (
                    f"<b>🟢 BUY ENTRY</b><br>"
                    f"Date: {pd.to_datetime(e_date).strftime('%Y-%m-%d')}<br>"
                    f"Price: ${e_price:.2f}<br>"
                    f"Trade Result: {'WIN (+)' if is_win else 'LOSS (-)'} ({ret:+.2f}%)"
                )
                if is_win:
                    win_entries_x.append(e_date)
                    win_entries_y.append(e_price)
                    win_entries_text.append(txt_e)
                else:
                    loss_entries_x.append(e_date)
                    loss_entries_y.append(e_price)
                    loss_entries_text.append(txt_e)
                    
            # Exit tick mark
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
                    
            # Dotted connector between entry and exit
            if e_date in df_index_set and x_date in df_index_set:
                if is_win:
                    win_lines_x.extend([e_date, x_date, None])
                    win_lines_y.extend([e_price, x_price, None])
                else:
                    loss_lines_x.extend([e_date, x_date, None])
                    loss_lines_y.extend([e_price, x_price, None])
                    
        # Add connecting lines
        if win_lines_x:
            fig.add_trace(go.Scatter(
                x=win_lines_x, y=win_lines_y, mode='lines',
                line=dict(color='rgba(0, 200, 5, 0.45)', width=1.5, dash='dot'),
                name="Profitable Trade Path", hoverinfo='skip'
            ), row=1, col=1)
            
        if loss_lines_x:
            fig.add_trace(go.Scatter(
                x=loss_lines_x, y=loss_lines_y, mode='lines',
                line=dict(color='rgba(255, 51, 58, 0.45)', width=1.5, dash='dot'),
                name="Unprofitable Trade Path", hoverinfo='skip'
            ), row=1, col=1)
            
        # Add Winning Ticks (Green)
        if win_entries_x:
            fig.add_trace(go.Scatter(
                x=win_entries_x, y=win_entries_y, mode='markers',
                marker=dict(symbol='triangle-up', size=11, color='#00C805', line=dict(width=1, color='#004D1A')),
                name="Win Entry (Buy)", hovertext=win_entries_text, hoverinfo='text'
            ), row=1, col=1)
            
        if win_exits_x:
            fig.add_trace(go.Scatter(
                x=win_exits_x, y=win_exits_y, mode='markers',
                marker=dict(symbol='triangle-down', size=11, color='#00C805', line=dict(width=1, color='#004D1A')),
                name="Win Exit (Sell)", hovertext=win_exits_text, hoverinfo='text'
            ), row=1, col=1)
            
        # Add Losing Ticks (Red)
        if loss_entries_x:
            fig.add_trace(go.Scatter(
                x=loss_entries_x, y=loss_entries_y, mode='markers',
                marker=dict(symbol='triangle-up', size=11, color='#FF333A', line=dict(width=1, color='#660000')),
                name="Loss Entry (Buy)", hovertext=loss_entries_text, hoverinfo='text'
            ), row=1, col=1)
            
        if loss_exits_x:
            fig.add_trace(go.Scatter(
                x=loss_exits_x, y=loss_exits_y, mode='markers',
                marker=dict(symbol='triangle-down', size=11, color='#FF333A', line=dict(width=1, color='#660000')),
                name="Loss Exit (Sell)", hovertext=loss_exits_text, hoverinfo='text'
            ), row=1, col=1)
            
    # 3. 2-Day CRSI Line
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
    
    # CRSI Reference Thresholds
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
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df_sp = pd.read_csv(
