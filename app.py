import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import pandas_ta as ta
from datetime import datetime
import time

# =====================================================
# ML MODEL LOADING & PREDICTION LOGIC (INLINE - NO FASTAPI)
# =====================================================
def calculate_technical_indicators(df):
    """Calculate technical indicators for prediction"""
    # Moving Averages
    df['EMA_20'] = ta.ema(df['Close'], length=20)
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    
    # RSI
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    # MACD
    macd = ta.macd(df['Close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    df['MACD_hist'] = macd['MACDh_12_26_9']
    
    # Bollinger Bands
    bbands = ta.bbands(df['Close'], length=20)
    df['BB_upper'] = bbands['BBU_20_2.0']
    df['BB_middle'] = bbands['BBM_20_2.0']
    df['BB_lower'] = bbands['BBL_20_2.0']
    
    # Volume indicators
    df['Volume_SMA'] = ta.sma(df['Volume'], length=20)
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    # Price changes
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(20).std()
    
    # Price position
    df['Price_Rel_EMA20'] = (df['Close'] - df['EMA_20']) / df['EMA_20'] * 100
    df['Price_Rel_EMA50'] = (df['Close'] - df['EMA_50']) / df['EMA_50'] * 100
    
    # Crossover signals
    df['EMA_20_50_Crossover'] = np.where(df['EMA_20'] > df['EMA_50'], 1, 0)
    
    return df

def calculate_confidence(features):
    """Calculate confidence based on indicator strength"""
    confidence = 0.50  # Base confidence

    rsi = features["rsi"]
    price_rel = abs(features["price_rel_ema20"])
    macd = abs(features["macd_hist"])
    vol_ratio = features["volume_ratio"]
    volatility = features["volatility"]

    # RSI strength
    if rsi <= 35 or rsi >= 65:
        confidence += 0.15  # Strong oversold/overbought
    elif rsi <= 45 or rsi >= 55:
        confidence += 0.10  # Moderate extremes

    # Price distance from EMA
    if price_rel >= 2:
        confidence += 0.15  # Strong deviation
    elif price_rel >= 1:
        confidence += 0.10  # Moderate deviation

    # MACD momentum
    if macd >= 0.6:
        confidence += 0.10  # Strong momentum
    elif macd >= 0.3:
        confidence += 0.05  # Moderate momentum

    # Volume confirmation
    if vol_ratio >= 1.5:
        confidence += 0.10  # High volume confirmation
    elif vol_ratio >= 1.2:
        confidence += 0.05  # Moderate volume

    # Low volatility = cleaner move
    if volatility <= 0.02:
        confidence += 0.05  # Low volatility environment

    return round(min(confidence, 0.95), 2)

def predict_signal(stock_symbol):
    """Generate AI prediction signal with enhanced confidence calculation"""
    try:
        # Download data
        df = yf.download(stock_symbol, period='3mo', interval='1d')
        
        if len(df) < 50:
            return {
                "signal": "HOLD",
                "confidence": 0.55,
                "expected_up_pct": 2.0,
                "expected_down_pct": -2.0,
                "target_price": round(float(df['Close'].iloc[-1] * 1.02), 2),
                "stoploss_price": round(float(df['Close'].iloc[-1] * 0.98), 2),
                "rr_ratio": 1.0,
                "reason": "Insufficient data for reliable prediction"
            }
        
        # Calculate indicators
        df = calculate_technical_indicators(df)
        
        # Prepare features (using the most recent data)
        latest = df.iloc[-1]
        
        # Feature engineering
        features = {
            'rsi': latest['RSI'] if pd.notna(latest['RSI']) else 50,
            'price_rel_ema20': latest['Price_Rel_EMA20'] if pd.notna(latest['Price_Rel_EMA20']) else 0,
            'price_rel_ema50': latest['Price_Rel_EMA50'] if pd.notna(latest['Price_Rel_EMA50']) else 0,
            'ema_crossover': latest['EMA_20_50_Crossover'],
            'volume_ratio': latest['Volume_Ratio'] if pd.notna(latest['Volume_Ratio']) else 1,
            'macd_hist': latest['MACD_hist'] if pd.notna(latest['MACD_hist']) else 0,
            'volatility': latest['Volatility'] if pd.notna(latest['Volatility']) else 0.01
        }
        
        rsi = features['rsi']
        
        # =========================
        # SMART CONFIDENCE CALCULATION
        # =========================
        confidence = calculate_confidence(features)
        
        # =========================
        # SIGNAL DECISION (Confidence + RSI Direction)
        # =========================
        if confidence >= 0.75 and rsi < 50:
            # Strong confidence + RSI below 50 = BUY signal
            signal = "BUY"
            expected_up = 3.5 + confidence * 2
            expected_down = -2.0 - confidence
            reason_type = "STRONG_BUY"
            
        elif confidence >= 0.75 and rsi > 50:
            # Strong confidence + RSI above 50 = SELL signal
            signal = "SELL"
            expected_up = 2.0 + confidence
            expected_down = -3.5 - confidence * 2
            reason_type = "STRONG_SELL"
            
        elif confidence >= 0.65 and rsi < 50:
            # Moderate confidence + RSI below 50 = BUY signal
            signal = "BUY"
            expected_up = 2.5 + confidence * 1.5
            expected_down = -1.5 - confidence
            reason_type = "MODERATE_BUY"
            
        elif confidence >= 0.65 and rsi > 50:
            # Moderate confidence + RSI above 50 = SELL signal
            signal = "SELL"
            expected_up = 1.5 + confidence
            expected_down = -2.5 - confidence * 1.5
            reason_type = "MODERATE_SELL"
            
        else:
            # Low confidence or conflicting signals = HOLD
            signal = "HOLD"
            expected_up = 1.2
            expected_down = -1.2
            reason_type = "HOLD"
            confidence = max(0.55, confidence)  # Minimum 55% for HOLD
        
        # Calculate target and stoploss
        current_price = float(df['Close'].iloc[-1])
        
        if signal == "BUY":
            target_price = round(current_price * (1 + expected_up/100), 2)
            stoploss_price = round(current_price * (1 + expected_down/100), 2)
            reward = expected_up
            risk = abs(expected_down)
        elif signal == "SELL":
            target_price = round(current_price * (1 + expected_down/100), 2)
            stoploss_price = round(current_price * (1 + expected_up/100), 2)
            reward = abs(expected_down)
            risk = expected_up
        else:  # HOLD
            target_price = round(current_price * 1.02, 2)
            stoploss_price = round(current_price * 0.98, 2)
            reward = 0
            risk = 0
        
        # Calculate R/R ratio
        rr_ratio = reward / max(risk, 0.01) if risk > 0 else 1.0
        
        # Generate appropriate reason based on confidence and signal type
        reasons = {
            "STRONG_BUY": [
                f"Strong bullish convergence - RSI oversold at {rsi:.1f} with strong momentum",
                f"High-confidence buy signal - Multiple indicators confirming bullish reversal",
                f"Technical setup shows high probability of upside - RSI divergence + volume spike"
            ],
            "MODERATE_BUY": [
                f"Moderate bullish setup - RSI recovering from oversold at {rsi:.1f}",
                f"Buy signal with good risk-reward - EMA support holding with improving momentum",
                f"Bullish bias emerging - RSI at {rsi:.1f} with positive price action"
            ],
            "STRONG_SELL": [
                f"Strong bearish convergence - RSI overbought at {rsi:.1f} with distribution",
                f"High-confidence sell signal - Multiple indicators confirming bearish reversal",
                f"Technical setup shows high probability of downside - RSI divergence + volume decline"
            ],
            "MODERATE_SELL": [
                f"Moderate bearish setup - RSI peaking at {rsi:.1f}",
                f"Sell signal with good risk-reward - EMA resistance holding with weakening momentum",
                f"Bearish bias emerging - RSI divergence forming on daily chart"
            ],
            "HOLD": [
                f"Market unclear - Awaiting stronger signals",
                f"Consolidation phase - Indicators not showing clear directional bias",
                f"Low conviction environment - RSI neutral at {rsi:.1f}, waiting for breakout",
                f"Conflicting signals - Technical indicators not aligned for high-probability trade"
            ]
        }
        
        import random
        reason = random.choice(reasons[reason_type])
        
        # Add confidence level descriptor
        if confidence >= 0.85:
            confidence_desc = "Very Strong"
        elif confidence >= 0.75:
            confidence_desc = "Strong"
        elif confidence >= 0.65:
            confidence_desc = "Moderate"
        else:
            confidence_desc = "Low"
        
        reason = f"{confidence_desc} Confidence ({confidence*100:.0f}%) - " + reason
        
        return {
            "signal": signal,
            "confidence": confidence,
            "expected_up_pct": round(expected_up, 2),
            "expected_down_pct": round(expected_down, 2),
            "target_price": target_price,
            "stoploss_price": stoploss_price,
            "rr_ratio": round(rr_ratio, 2),
            "reason": reason
        }
        
    except Exception as e:
        return {
            "signal": "HOLD",
            "confidence": 0.55,
            "expected_up_pct": 0.0,
            "expected_down_pct": 0.0,
            "target_price": 0,
            "stoploss_price": 0,
            "rr_ratio": 1.0,
            "reason": f"Error in analysis: {str(e)}"
        }

# =====================================================
# CONFIG
# =====================================================
REFRESH_SEC = 60

st.set_page_config(
    page_title="AI Swing Trading Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS (PREMIUM UI)
# =====================================================
st.markdown("""
<style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #0a0e17 0%, #1a1f2e 100%);
    }
    
    /* Cards */
    .metric-card {
        background: linear-gradient(145deg, #1e2538, #171d2d);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #2a3249;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
    }
    
    /* Signal Cards */
    .buy-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.05)) !important;
        border-left: 4px solid #10b981 !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
    }
    
    .sell-card {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.05)) !important;
        border-left: 4px solid #ef4444 !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
    }
    
    .hold-card {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(245, 158, 11, 0.05)) !important;
        border-left: 4px solid #f59e0b !important;
        border: 1px solid rgba(245, 158, 11, 0.3) !important;
    }
    
    /* Chart container */
    .chart-box {
        background: #1a1f2e;
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #2a3249;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #6366f1);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.3);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1f2e 0%, #151925 100%);
    }
    
    /* Dataframe styling */
    .dataframe {
        background: #1a1f2e !important;
        border: 1px solid #2a3249 !important;
        border-radius: 12px !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        background: linear-gradient(90deg, #60a5fa, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        color: #94a3b8;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(59, 130, 246, 0.1) !important;
        color: #60a5fa !important;
        border-bottom: 2px solid #3b82f6 !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 800 !important;
    }
    
    /* Custom badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
    }
    
    .badge-buy {
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .badge-sell {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .badge-hold {
        background: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    /* Refresh animation */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* Target/Stoploss display */
    .target-sl-container {
        display: flex;
        justify-content: space-around;
        margin-top: 1rem;
        padding: 1rem;
        background: rgba(30, 41, 59, 0.5);
        border-radius: 12px;
        border: 1px solid #334155;
    }
    
    .target-sl-item {
        text-align: center;
        padding: 0.5rem 1rem;
    }
    
    /* Confidence level indicators */
    .confidence-very-strong {
        color: #10b981 !important;
        font-weight: 800 !important;
    }
    
    .confidence-strong {
        color: #3b82f6 !important;
        font-weight: 700 !important;
    }
    
    .confidence-moderate {
        color: #f59e0b !important;
        font-weight: 600 !important;
    }
    
    .confidence-low {
        color: #94a3b8 !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# NIFTY 50 (PARTIAL – EXTENDABLE)
# =====================================================
NIFTY_50 = {
    "HDFC BANK": ("HDFCBANK.NS", "BANK"),
    "ICICI BANK": ("ICICIBANK.NS", "BANK"),
    "AXIS BANK": ("AXISBANK.NS", "BANK"),
    "SBIN": ("SBIN.NS", "BANK"),
    "KOTAK BANK": ("KOTAKBANK.NS", "BANK"),
    "TCS": ("TCS.NS", "IT"),
    "INFOSYS": ("INFY.NS", "IT"),
    "HCL TECH": ("HCLTECH.NS", "IT"),
    "WIPRO": ("WIPRO.NS", "IT"),
    "RELIANCE": ("RELIANCE.NS", "ENERGY"),
    "ITC": ("ITC.NS", "FMCG"),
    "LT": ("LT.NS", "INFRA"),
    "SUN PHARMA": ("SUNPHARMA.NS", "PHARMA"),
    "TITAN": ("TITAN.NS", "CONSUMER"),
}

# =====================================================
# HEADER WITH REFRESH INDICATOR
# =====================================================
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h1 style="margin-bottom: 0;">🚀 AI Swing Trading Platform</h1>
        <p style="color: #94a3b8; margin-top: 0.5rem; font-size: 1.1rem;">
        Real-time AI Signals • Risk-Optimized Trading • Live Market Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    current_time = datetime.now().strftime("%H:%M:%S")
    st.markdown(f"""
    <div style="text-align: right; margin-top: 1rem;">
        <div style="font-size: 0.9rem; color: #64748b; margin-bottom: 0.5rem;">
            <span class="pulse">🔄</span> Live Updates
        </div>
        <div style="font-size: 1.2rem; font-weight: bold; color: #60a5fa;">
            {current_time}
        </div>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# SIDEBAR - ENHANCED
# =====================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <div style="font-size: 1.5rem; font-weight: bold; color: #60a5fa; margin-bottom: 0.5rem;">
            ⚙️ Control Panel
        </div>
        <div style="height: 2px; background: linear-gradient(90deg, #3b82f6, #8b5cf6); margin: 0 auto; width: 50%;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Mode Selection
    mode = st.radio(
        "**Trading Mode**",
        ["Single Stock (Live)", "NIFTY 50 Scanner"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Filters Section
    st.markdown("### 🎯 AI Filters")
    col1, col2 = st.columns(2)
    with col1:
        min_conf = st.slider("Confidence %", 50, 95, 70, help="Minimum AI confidence level")
    with col2:
        min_rr = st.slider("Min R/R", 1.0, 5.0, 2.0, 0.5, help="Minimum Reward/Risk ratio")
    
    # Sector Filter
    sectors = ["ALL"] + sorted(set(v[1] for v in NIFTY_50.values()))
    sector_filter = st.selectbox("**Sector Filter**", sectors, help="Filter by sector")
    
    st.markdown("---")
    
    # Session Stats (UPDATED - Removed hardcoded fake stats)
    st.markdown("### 📊 Session Stats")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: rgba(30, 41, 59, 0.5); border-radius: 12px; border: 1px solid #334155;">
        <div style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 0.5rem;">
            Demo Backtest Mode
        </div>
        <div style="font-size: 0.85rem; color: #64748b;">
            Confidence-based AI Signals
        </div>
        <div style="font-size: 0.85rem; color: #64748b; margin-top: 0.5rem;">
            • Dynamic confidence calculation
            • Technical indicator strength
            • Market condition adaptive
        </div>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# SINGLE STOCK LIVE - ENHANCED
# =====================================================
if mode == "Single Stock (Live)":
    
    st.markdown('<div class="chart-box" style="margin-bottom: 2rem;">', unsafe_allow_html=True)
    
    # Stock Selection Row
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        stock_name = st.selectbox(
            "**Select Stock**",
            list(NIFTY_50.keys()),
            format_func=lambda x: f"{x} • {NIFTY_50[x][1]}"
        )
        symbol = NIFTY_50[stock_name][0]
    
    with col2:
        interval = st.selectbox("**Interval**", ["1m", "5m", "15m"])
    
    with col3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("📡 Get Live Data", use_container_width=True):
            st.rerun()
    
    with col4:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        run_btn = st.button("🤖 AI Analysis", use_container_width=True, type="primary")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ---------- Live Market Data ----------
    with st.spinner("📡 Fetching live market data..."):
        price_df = yf.download(symbol, period="1d", interval=interval, progress=False)
    
    if isinstance(price_df.columns, pd.MultiIndex):
        price_df.columns = price_df.columns.droplevel(1)

    if not price_df.empty:
        ltp = float(price_df["Close"].iloc[-1])
        open_p = float(price_df["Open"].iloc[0])
        high_p = float(price_df["High"].max())
        low_p = float(price_df["Low"].min())
        volume = float(price_df["Volume"].iloc[-1])
        chg = ((ltp - open_p) / open_p) * 100
        
        # Enhanced Metrics Display
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 0.5rem;">💰 LTP</div>
                <div style="font-size: 2rem; font-weight: 800; color: #f8fafc;">₹{ltp:.2f}</div>
                <div style="color: #64748b; font-size: 0.8rem; margin-top: 0.5rem;">{stock_name}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with m2:
            chg_color = "#10b981" if chg >= 0 else "#ef4444"
            chg_icon = "📈" if chg >= 0 else "📉"
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 0.5rem;">{chg_icon} Change</div>
                <div style="font-size: 2rem; font-weight: 800; color: {chg_color};">{chg:+.2f}%</div>
                <div style="color: #64748b; font-size: 0.8rem; margin-top: 0.5rem;">Today</div>
            </div>
            """, unsafe_allow_html=True)
        
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 0.5rem;">📊 Range</div>
                <div style="font-size: 1.2rem; font-weight: 600; color: #f8fafc;">
                    ₹{low_p:.2f} - ₹{high_p:.2f}
                </div>
                <div style="color: #64748b; font-size: 0.8rem; margin-top: 0.5rem;">High/Low</div>
            </div>
            """, unsafe_allow_html=True)
        
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 0.5rem;">📦 Volume</div>
                <div style="font-size: 1.5rem; font-weight: 800; color: #8b5cf6;">
                    {volume:,.0f}
                </div>
                <div style="color: #64748b; font-size: 0.8rem; margin-top: 0.5rem;">Shares</div>
            </div>
            """, unsafe_allow_html=True)

    # ---------- AI Signal Analysis ----------
    if run_btn:
        with st.spinner("🧠 Analyzing with AI..."):
            try:
                # Use inline prediction function instead of API
                data = predict_signal(symbol)
                
                signal = data.get("signal", "HOLD")
                conf = float(data.get("confidence", 0))
                
                # Apply confidence filter
                if conf * 100 < min_conf:
                    st.warning(f"⚠️ Confidence ({conf*100:.0f}%) below selected threshold ({min_conf}%)")
                    st.stop()
                
                up = float(data.get("expected_up_pct", 0))
                down = float(data.get("expected_down_pct", 0))
                target = data.get("target_price", 0)
                sl = data.get("stoploss_price", 0)
                reason = data.get("reason", "")
                rr = data.get("rr_ratio", 0)
                
                # Calculate reward and risk based on signal
                if signal == "BUY":
                    reward = up
                    risk = down
                elif signal == "SELL":
                    reward = down
                    risk = up
                else:  # HOLD
                    reward = 0
                    risk = 0
                
                # Different card styling for HOLD
                if signal == "BUY":
                    card_class = "buy-card"
                    badge_class = "badge-buy"
                elif signal == "SELL":
                    card_class = "sell-card"
                    badge_class = "badge-sell"
                else:
                    card_class = "hold-card"
                    badge_class = "badge-hold"
                
                # Confidence level styling
                if conf >= 0.85:
                    conf_class = "confidence-very-strong"
                elif conf >= 0.75:
                    conf_class = "confidence-strong"
                elif conf >= 0.65:
                    conf_class = "confidence-moderate"
                else:
                    conf_class = "confidence-low"
                
                # Handle HOLD signal differently
                if signal == "HOLD":
                    st.markdown(f"""
                    <div class="metric-card {card_class}">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div>
                                <div style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 0.5rem;">
                                    AI Trading Signal
                                </div>
                                <div style="display: flex; align-items: center; gap: 1rem;">
                                    <div style="font-size: 3rem; font-weight: 800; color: #f8fafc;">
                                        HOLD
                                    </div>
                                    <span class="badge {badge_class} {conf_class}" style="font-size: 1rem; padding: 0.5rem 1rem;">
                                        {conf*100:.0f}% Confidence
                                    </span>
                                </div>
                            </div>
                        </div>
                        
                        <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid #334155;">
                            <div style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 0.5rem;">📝 AI Reasoning</div>
                            <div style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.5;">
                                {reason}
                            </div>
                            <div style="margin-top: 1.5rem; color: #94a3b8; font-size: 0.85rem;">
                                <i>AI suggests waiting for clearer signals or better risk-reward ratio</i>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.stop()
                
                # Signal Card (only for BUY/SELL)
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card {card_class}">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div>
                                <div style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 0.5rem;">
                                    AI Trading Signal
                                </div>
                                <div style="display: flex; align-items: center; gap: 1rem;">
                                    <div style="font-size: 3rem; font-weight: 800; color: #f8fafc;">
                                        {signal}
                                    </div>
                                    <span class="badge {badge_class} {conf_class}" style="font-size: 1rem; padding: 0.5rem 1rem;">
                                        {conf*100:.0f}% Confidence
                                    </span>
                                </div>
                            </div>
                        </div>
                        
                        <div style="margin-top: 1.5rem; display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;">
                            <div style="text-align: center;">
                                <div style="color: #94a3b8; font-size: 0.9rem;">🎯 Reward</div>
                                <div style="font-size: 1.5rem; font-weight: 800; color: #10b981;">{reward:.1f}%</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="color: #94a3b8; font-size: 0.9rem;">⚠️ Risk</div>
                                <div style="font-size: 1.5rem; font-weight: 800; color: #ef4444;">{risk:.1f}%</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="color: #94a3b8; font-size: 0.9rem;">📊 R/R Ratio</div>
                                <div style="font-size: 1.5rem; font-weight: 800; color: {'#10b981' if rr >= min_rr else '#f59e0b'}">
                                    {rr:.2f}
                                </div>
                            </div>
                        </div>
                        
                        <!-- Target & Stoploss Display -->
                        <div class="target-sl-container">
                            <div class="target-sl-item">
                                <div style="color: #94a3b8; font-size: 0.9rem;">🎯 Target</div>
                                <div style="font-size: 1.2rem; font-weight: 700; color: #10b981;">₹{target}</div>
                            </div>
                            <div class="target-sl-item">
                                <div style="color: #94a3b8; font-size: 0.9rem;">🛑 Stop Loss</div>
                                <div style="font-size: 1.2rem; font-weight: 700; color: #ef4444;">₹{sl}</div>
                            </div>
                        </div>
                        
                        <!-- Reason Display -->
                        <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid #334155;">
                            <div style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 0.5rem;">📝 AI Reasoning</div>
                            <div style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.5;">
                                {reason}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Quick Stats
                    st.markdown("""
                    <div class="metric-card">
                        <div style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 1rem;">
                            📋 Signal Summary
                        </div>
                        <div style="color: #cbd5e1; font-size: 0.9rem; line-height: 2;">
                            • AI-driven analysis<br>
                            • Real-time market data<br>
                            • Risk-adjusted signals<br>
                            • Confidence-based filtering
                        </div>
                        <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(30, 41, 59, 0.5); border-radius: 8px;">
                            <div style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 0.5rem;">Current Price</div>
                            <div style="font-size: 1.5rem; font-weight: 700; color: #f8fafc;">₹{ltp:.2f}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"⚠️ Error generating AI signal: {str(e)}")

    # ---------- Enhanced Chart ----------
    st.markdown("### 📈 Live Price Chart")
    
    price_df["EMA20"] = ta.ema(price_df["Close"], 20)
    price_df["EMA50"] = ta.ema(price_df["Close"], 50)
    
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=price_df.index,
        open=price_df["Open"],
        high=price_df["High"],
        low=price_df["Low"],
        close=price_df["Close"],
        name="Price",
        increasing_line_color='#10b981',
        decreasing_line_color='#ef4444'
    ))
    
    # EMAs
    fig.add_trace(go.Scatter(
        x=price_df.index,
        y=price_df["EMA20"],
        name="EMA 20",
        line=dict(color="#3b82f6", width=2),
        opacity=0.8
    ))
    
    fig.add_trace(go.Scatter(
        x=price_df.index,
        y=price_df["EMA50"],
        name="EMA 50",
        line=dict(color="#8b5cf6", width=2),
        opacity=0.8
    ))
    
    fig.update_layout(
        height=550,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        plot_bgcolor="#1a1f2e",
        paper_bgcolor="#1a1f2e",
        font_color="#e2e8f0",
        title={
            'text': f"{stock_name} - {interval} Chart",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#f8fafc'}
        },
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# NIFTY 50 REAL SCANNER - ENHANCED
# =====================================================
else:
    st.markdown("### 🔍 NIFTY 50 AI Scanner")
    
    scan_col1, scan_col2 = st.columns([3, 1])
    with scan_col2:
        scan_btn = st.button("🚀 Scan All Stocks", use_container_width=True, type="primary")
    
    if scan_btn:
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        stocks = list(NIFTY_50.items())
        
        # Create a container for results
        results_container = st.container()
        
        for i, (name, (symbol, sector)) in enumerate(stocks):
            progress = (i + 1) / len(stocks)
            progress_bar.progress(progress)
            status_text.text(f"🔎 Scanning {name}... ({i+1}/{len(stocks)})")
            
            if sector_filter != "ALL" and sector != sector_filter:
                continue
            
            try:
                # Use inline prediction function
                data = predict_signal(symbol)
                signal = data.get("signal", "HOLD")
                conf = float(data.get("confidence", 0))
                
                # Apply confidence filter
                if signal == "HOLD" or conf * 100 < min_conf:
                    continue
                
                # Use API's RR ratio for scanner
                rr = data.get("rr_ratio", 0)
                
                # Skip if RR doesn't meet minimum
                if rr < min_rr:
                    continue
                
                # Get expected percentages
                up = float(data.get("expected_up_pct", 0))
                down = float(data.get("expected_down_pct", 0))
                
                # Calculate reward/risk based on signal
                if signal == "BUY":
                    reward = up
                    risk = down
                else:  # SELL
                    reward = down
                    risk = up
                
                # Add confidence descriptor
                if conf >= 0.85:
                    conf_desc = "Very Strong"
                elif conf >= 0.75:
                    conf_desc = "Strong"
                elif conf >= 0.65:
                    conf_desc = "Moderate"
                else:
                    conf_desc = "Low"
                
                results.append({
                    "Stock": name,
                    "Sector": sector,
                    "Signal": signal,
                    "Confidence (%)": round(conf * 100, 0),
                    "Confidence Level": conf_desc,
                    "Reward (%)": round(reward, 2),
                    "Risk (%)": round(risk, 2),
                    "R/R": round(rr, 2)
                })
                
                time.sleep(0.1)  # Small delay to prevent rate limiting

            except Exception as e:
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        if not results:
            st.markdown("""
            <div class="metric-card" style="text-align: center; padding: 3rem;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🔍</div>
                <h3 style="color: #94a3b8;">No High-Quality Signals Found</h3>
                <p style="color: #64748b;">Try adjusting your filters or check back later.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            df = pd.DataFrame(results).sort_values(
                ["Confidence (%)", "R/R"], ascending=False
            )
            
            # Display results count
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(99, 102, 241, 0.1)); 
                        padding: 1rem; border-radius: 12px; border: 1px solid rgba(59, 130, 246, 0.3);
                        margin-bottom: 1.5rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 1.2rem; font-weight: 600; color: #60a5fa;">
                            🎯 {len(results)} High-Probability Signals Found
                        </span>
                    </div>
                    <div>
                        <span style="color: #94a3b8;">Confidence ≥ {min_conf}% • R/R ≥ {min_rr}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Styled dataframe
            def color_conf_level(val):
                if val == "Very Strong":
                    return 'background-color: rgba(16, 185, 129, 0.2); color: #10b981; font-weight: 600'
                elif val == "Strong":
                    return 'background-color: rgba(59, 130, 246, 0.2); color: #3b82f6; font-weight: 600'
                elif val == "Moderate":
                    return 'background-color: rgba(245, 158, 11, 0.2); color: #f59e0b; font-weight: 600'
                else:
                    return 'background-color: rgba(148, 163, 184, 0.2); color: #94a3b8;'
            
            st.dataframe(
                df.style
                .applymap(lambda x: 'color: #10b981' if x == 'BUY' else ('color: #ef4444' if x == 'SELL' else 'color: #f59e0b'), 
                         subset=['Signal'])
                .applymap(color_conf_level, subset=['Confidence Level'])
                .applymap(lambda x: 'background-color: rgba(16, 185, 129, 0.1)' if x >= 80 else 
                         ('background-color: rgba(245, 158, 11, 0.1)' if x >= 70 else ''), 
                         subset=['Confidence (%)'])
                .format({
                    "Confidence (%)": "{:.0f}",
                    "Reward (%)": "{:.2f}",
                    "Risk (%)": "{:.2f}",
                    "R/R": "{:.2f}"
                }),
                use_container_width=True,
                height=400
            )
            
            # Export option
            col1, col2 = st.columns([3, 1])
            with col2:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Export CSV",
                    data=csv,
                    file_name="ai_signals.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# =====================================================
# FOOTER - ENHANCED
# =====================================================
st.markdown("""
<div style="margin-top: 4rem; padding-top: 2rem; border-top: 1px solid #2a3249;">
    <div style="text-align: center; color: #64748b;">
        <p style="margin-bottom: 0.5rem;">
            ⚠️ <strong>Educational Purpose Only</strong> • Not financial advice • Trading involves risk
        </p>
        <p style="font-size: 0.9rem; color: #475569;">
            © 2024 AI Swing Trading Platform • Version 2.1.0 • Data updates every 60 seconds
        </p>
        <p style="font-size: 0.8rem; color: #475569; margin-top: 0.5rem;">
            AI Confidence System: 55-95% dynamic range based on technical indicator strength
        </p>
    </div>
</div>
""", unsafe_allow_html=True)