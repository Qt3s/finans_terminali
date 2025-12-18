"""
Profesyonel Finans Terminali v3.0 - Buffett Edition
Warren Buffett'ın bilanço odaklı yatırım felsefesini kripto ve hisse piyasalarına entegre eden
modüler, yüksek performanslı Streamlit terminali.

Özellikler:
- On-Chain Bilanço Analizi (DeFiLlama API)
- Buffett Finansal Sağlık Skoru (1-10)
- EMA Teknik İndikatörleri (20, 50, 200)
- Kripto + Hisse Senedi Terminalleri
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta

# ==================== SAYFA KONFİGÜRASYONU ====================

st.set_page_config(
    page_title="Finans Terminali - Buffett Edition",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: bold;
    }
    [data-testid="stMetricDelta"] {
        font-size: 1rem;
    }
    .buffett-score {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
    }
    @media (max-width: 768px) {
        [data-testid="stMetricValue"] {
            font-size: 1.3rem;
        }
        .main .block-container {
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)


# ==================== SABİTLER ====================

EXCHANGE_CONFIGS = [
    {'name': 'kucoin', 'class': 'kucoin', 'options': {'enableRateLimit': True}, 'symbol_map': {}},
    {'name': 'kraken', 'class': 'kraken', 'options': {'enableRateLimit': True}, 'symbol_map': {}},
]

CRYPTO_SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT", "AVAX/USDT", "MATIC/USDT"]
TIMEFRAMES = {"1 Saat": "1h", "4 Saat": "4h", "1 Gün": "1d", "1 Hafta": "1w"}

# DeFiLlama protokol listesi
DEFI_PROTOCOLS = {
    "Aave": "aave",
    "Uniswap": "uniswap",
    "Lido": "lido",
    "MakerDAO": "makerdao",
    "Curve": "curve-dex",
    "Compound": "compound-finance",
    "Convex": "convex-finance",
    "Balancer": "balancer",
    "SushiSwap": "sushiswap",
    "PancakeSwap": "pancakeswap",
}


# ==================== VERİ ÇEKİCİ FONKSİYONLAR ====================

def get_exchange_instance(config):
    """Borsa instance'ı oluşturur."""
    import ccxt
    exchange_class = getattr(ccxt, config['class'])
    return exchange_class(config['options'])


@st.cache_data(ttl=600, show_spinner=False)
def fetch_crypto_ticker(symbol: str):
    """Kripto fiyat bilgisi (fallback mekanizması)."""
    import ccxt
    errors = []
    
    for config in EXCHANGE_CONFIGS:
        try:
            exchange = get_exchange_instance(config)
            ticker = exchange.fetch_ticker(symbol)
            return ticker, None, config['name']
        except Exception as e:
            errors.append(f"{config['name']}: {str(e)}")
            continue
    
    return None, " | ".join(errors), None


@st.cache_data(ttl=600, show_spinner=False)
def fetch_crypto_ohlcv(symbol: str, timeframe: str, limit: int = 200):
    """Kripto OHLCV verisi + EMA hesaplama."""
    import ccxt
    errors = []
    
    for config in EXCHANGE_CONFIGS:
        try:
            exchange = get_exchange_instance(config)
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # EMA hesaplama
            df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['EMA_200'] = df['close'].ewm(span=200, adjust=False).mean()
            
            return df, None, config['name']
        except Exception as e:
            errors.append(f"{config['name']}: {str(e)}")
            continue
    
    return None, " | ".join(errors), None


@st.cache_data(ttl=900, show_spinner=False)
def fetch_stock_data(symbol: str, period: str = "6mo"):
    """Yahoo Finance'den hisse verisi."""
    import time
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                return None, f"'{symbol}' için veri bulunamadı."
            
            # EMA hesaplama
            hist['EMA_20'] = hist['Close'].ewm(span=20, adjust=False).mean()
            hist['EMA_50'] = hist['Close'].ewm(span=50, adjust=False).mean()
            
            return hist, None
        except Exception as e:
            error_msg = str(e).lower()
            if "rate" in error_msg or "too many" in error_msg:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
            return None, str(e)
    
    return None, "Rate limit aşıldı."


@st.cache_data(ttl=600, show_spinner=False)
def fetch_defillama_protocol(protocol_slug: str):
    """DeFiLlama'dan protokol verisi çeker."""
    try:
        url = f"https://api.llama.fi/protocol/{protocol_slug}"
        response = requests.get(url, timeout=15)
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Hatası: {response.status_code}"
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=600, show_spinner=False)
def fetch_defillama_treasury(protocol_slug: str):
    """DeFiLlama'dan treasury verisi çeker."""
    try:
        url = f"https://api.llama.fi/treasury/{protocol_slug}"
        response = requests.get(url, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            return data, None
        else:
            return None, f"Treasury verisi yok"
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_ethereum_data():
    """Ethereum ağ durumu."""
    try:
        from web3 import Web3
        
        rpc_endpoints = [
            "https://cloudflare-eth.com",
            "https://eth.llamarpc.com",
            "https://rpc.ankr.com/eth",
        ]
        
        for rpc_url in rpc_endpoints:
            try:
                w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={'timeout': 10}))
                if w3.is_connected():
                    return {
                        'block_number': w3.eth.block_number,
                        'gas_price_gwei': round(w3.eth.gas_price / 1e9, 2),
                        'rpc_used': rpc_url
                    }, None
            except:
                continue
        
        return None, "RPC bağlantısı başarısız."
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=600, show_spinner=False)
def fetch_macro_data():
    """Makro ekonomi verileri: DXY, Bonds, Gold, VIX."""
    import yfinance as yf
    
    symbols = {
        'DXY': 'DX-Y.NYB',
        'US10Y': '^TNX',
        'Gold': 'GC=F',
        'VIX': '^VIX'
    }
    
    results = {}
    
    for name, symbol in symbols.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='5d')
            
            if not hist.empty:
                last = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2] if len(hist) > 1 else last
                change = ((last - prev) / prev) * 100 if prev != 0 else 0
                
                results[name] = {
                    'value': last,
                    'change': change,
                    'history': hist
                }
            else:
                results[name] = None
        except Exception as e:
            results[name] = None
    
    return results


@st.cache_data(ttl=600, show_spinner=False)
def fetch_correlation_data(crypto_symbol: str = "BTC-USD", days: int = 60):
    """DXY ve Kripto arasındaki korelasyonu hesaplar."""
    import yfinance as yf
    import numpy as np
    
    try:
        # Daha uzun periyot çek (hafta sonları için)
        dxy = yf.Ticker('DX-Y.NYB')
        crypto = yf.Ticker(crypto_symbol)
        
        dxy_hist = dxy.history(period=f'{days}d')
        crypto_hist = crypto.history(period=f'{days}d')
        
        if dxy_hist.empty or crypto_hist.empty:
            return None, "Veri yetersiz"
        
        # Tarihleri sadece güne normalize et (saat bilgisini kaldır)
        dxy_hist.index = dxy_hist.index.normalize()
        crypto_hist.index = crypto_hist.index.normalize()
        
        # Günlük kapanış getirilerini hesapla
        dxy_returns = dxy_hist['Close'].pct_change().dropna()
        crypto_returns = crypto_hist['Close'].pct_change().dropna()
        
        # Ortak tarihleri bul (sadece DXY'nin işlem gördüğü günler)
        common_dates = dxy_returns.index.intersection(crypto_returns.index)
        
        if len(common_dates) < 5:
            # Alternatif: resample ile haftalık korelasyon
            dxy_weekly = dxy_hist['Close'].resample('W').last().pct_change().dropna()
            crypto_weekly = crypto_hist['Close'].resample('W').last().pct_change().dropna()
            
            common_weeks = dxy_weekly.index.intersection(crypto_weekly.index)
            
            if len(common_weeks) < 3:
                return None, "Yeterli veri yok"
            
            dxy_aligned = dxy_weekly.loc[common_weeks]
            crypto_aligned = crypto_weekly.loc[common_weeks]
            period_label = f"{len(common_weeks)} hafta"
        else:
            dxy_aligned = dxy_returns.loc[common_dates]
            crypto_aligned = crypto_returns.loc[common_dates]
            period_label = f"{len(common_dates)} gün"
        
        correlation = np.corrcoef(dxy_aligned, crypto_aligned)[0, 1]
        
        return {
            'correlation': correlation,
            'dxy_data': dxy_hist,
            'crypto_data': crypto_hist,
            'days': period_label
        }, None
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=600, show_spinner=False)
def fetch_protocol_revenue(protocol_slug: str):
    """DeFiLlama'dan protokol gelir verisini çeker."""
    try:
        url = f"https://api.llama.fi/summary/fees/{protocol_slug}?dataType=dailyRevenue"
        response = requests.get(url, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            # Son 30 günlük toplam gelir
            total_30d = data.get('total30d', 0)
            total_24h = data.get('total24h', 0)
            return {
                'revenue_30d': total_30d,
                'revenue_24h': total_24h
            }, None
        else:
            return None, "Gelir verisi yok"
    except Exception as e:
        return None, str(e)


# ==================== BUFFETT SKOR HESAPLAMA ====================

def calculate_buffett_score(mcap: float, tvl: float, treasury_data: dict = None):
    """
    Warren Buffett tarzı finansal sağlık skoru (1-10).
    
    Kriterler:
    - Mcap/TVL Oranı: Düşük = iyi (F/K benzeri)
    - Treasury Stablecoin %: Yüksek = güvenli
    - Treasury Çeşitliliği: Dış varlık var mı?
    """
    score = 10
    details = []
    
    # Tip güvenliği
    try:
        tvl = float(tvl) if tvl else 0.0
    except (TypeError, ValueError):
        tvl = 0.0
    
    try:
        mcap = float(mcap) if mcap else 0.0
    except (TypeError, ValueError):
        mcap = 0.0
    
    # 1. Mcap/TVL Oranı
    if tvl > 0:
        mcap_tvl = mcap / tvl if mcap > 0 else 0
        
        if mcap_tvl > 5:
            score -= 4
            details.append(f"🔴 Mcap/TVL çok yüksek ({mcap_tvl:.2f})")
        elif mcap_tvl > 3:
            score -= 2
            details.append(f"🟡 Mcap/TVL yüksek ({mcap_tvl:.2f})")
        elif mcap_tvl > 1:
            score -= 1
            details.append(f"🟢 Mcap/TVL makul ({mcap_tvl:.2f})")
        else:
            details.append(f"🟢 Mcap/TVL düşük - potansiyel ucuz ({mcap_tvl:.2f})")
    else:
        score -= 2
        details.append("⚪ TVL verisi yok")
    
    # 2. Treasury Analizi
    if treasury_data:
        try:
            total_treasury = float(treasury_data.get('tvl', 0) or 0)
        except (TypeError, ValueError):
            total_treasury = 0.0
        
        if total_treasury > 100_000_000:  # 100M+
            details.append(f"🟢 Güçlü hazine (${total_treasury/1e6:.0f}M)")
        elif total_treasury > 10_000_000:  # 10M+
            score -= 1
            details.append(f"🟡 Orta hazine (${total_treasury/1e6:.0f}M)")
        elif total_treasury > 0:
            score -= 2
            details.append(f"🔴 Zayıf hazine (${total_treasury/1e6:.0f}M)")
        else:
            score -= 1
            details.append("⚪ Hazine tutarı bilinmiyor")
    else:
        score -= 1
        details.append("⚪ Hazine verisi yok")
    
    # 3. TVL Trend (basit kontrol)
    if tvl > 1_000_000_000:  # 1B+
        details.append("🟢 Yüksek TVL ($1B+)")
    elif tvl > 100_000_000:  # 100M+
        details.append("🟡 Orta TVL")
    elif tvl > 0:
        score -= 1
        details.append("🔴 Düşük TVL")
    
    return max(1, min(10, score)), details


# ==================== SAYFA FONKSİYONLARI ====================

def render_dashboard():
    """Ana Dashboard - Piyasa Özeti"""
    st.title("🏠 Piyasa Özeti")
    st.caption(f"Son güncelleme: {datetime.now().strftime('%H:%M:%S')}")
    st.divider()
    
    # Kripto Özet
    st.subheader("🪙 Kripto Piyasası")
    cols = st.columns(4)
    
    crypto_list = [("BTC/USDT", "Bitcoin"), ("ETH/USDT", "Ethereum"), ("SOL/USDT", "Solana"), ("BNB/USDT", "BNB")]
    
    for col, (symbol, name) in zip(cols, crypto_list):
        with col:
            data, error, _ = fetch_crypto_ticker(symbol)
            if data:
                st.metric(
                    label=name,
                    value=f"${data.get('last', 0):,.0f}" if data.get('last', 0) > 100 else f"${data.get('last', 0):,.2f}",
                    delta=f"{data.get('percentage', 0):+.2f}%"
                )
            else:
                st.metric(label=name, value="—")
    
    st.divider()
    
    # Hisse Özet
    st.subheader("📈 ABD Hisse Piyasası")
    cols = st.columns(4)
    
    stock_list = [("AAPL", "Apple"), ("GOOGL", "Google"), ("MSFT", "Microsoft"), ("NVDA", "NVIDIA")]
    
    for col, (symbol, name) in zip(cols, stock_list):
        with col:
            data, error = fetch_stock_data(symbol, "5d")
            if data is not None and not data.empty:
                last = data['Close'].iloc[-1]
                prev = data['Close'].iloc[-2] if len(data) > 1 else last
                change = ((last - prev) / prev) * 100
                st.metric(label=name, value=f"${last:,.2f}", delta=f"{change:+.2f}%")
            else:
                st.metric(label=name, value="—")
    
    st.divider()
    
    # Ethereum Ağ
    st.subheader("⛓️ Ethereum Ağı")
    col1, col2 = st.columns(2)
    
    eth_data, _ = fetch_ethereum_data()
    if eth_data:
        with col1:
            st.metric("📦 Son Blok", f"{eth_data['block_number']:,}")
        with col2:
            gas = eth_data['gas_price_gwei']
            status = "🟢" if gas < 20 else "🟡" if gas < 50 else "🔴"
            st.metric(f"⛽ Gas {status}", f"{gas} Gwei")


def render_crypto_page():
    """Kripto Terminal Sayfası"""
    st.title("🪙 Kripto Analiz Terminali")
    
    # Filtreler
    col1, col2, col3 = st.columns([2, 2, 4])
    with col1:
        selected_crypto = st.selectbox("Parite", CRYPTO_SYMBOLS, key='crypto_select')
    with col2:
        selected_tf = st.selectbox("Periyot", list(TIMEFRAMES.keys()), index=1, key='tf_select')
    
    st.divider()
    
    # Fiyat Metrikleri
    ticker, ticker_err, exchange = fetch_crypto_ticker(selected_crypto)
    
    if ticker:
        st.caption(f"📡 Kaynak: {exchange.upper()}")
        cols = st.columns(4)
        
        with cols[0]:
            st.metric("💰 Fiyat", f"${ticker.get('last', 0):,.2f}", f"{ticker.get('percentage', 0):+.2f}%")
        with cols[1]:
            st.metric("📈 24s Yüksek", f"${ticker.get('high', 0):,.2f}")
        with cols[2]:
            st.metric("📉 24s Düşük", f"${ticker.get('low', 0):,.2f}")
        with cols[3]:
            vol = ticker.get('quoteVolume', 0) or 0
            st.metric("📊 24s Hacim", f"${vol/1e6:,.1f}M")
    else:
        st.error(f"Fiyat alınamadı: {ticker_err}")
    
    st.divider()
    
    # Grafik + EMA
    st.subheader("📊 Fiyat Grafiği + EMA İndikatörleri")
    
    ohlcv, ohlcv_err, _ = fetch_crypto_ohlcv(selected_crypto, TIMEFRAMES[selected_tf])
    
    if ohlcv is not None and not ohlcv.empty:
        fig = go.Figure()
        
        # Mum grafiği
        fig.add_trace(go.Candlestick(
            x=ohlcv['timestamp'],
            open=ohlcv['open'],
            high=ohlcv['high'],
            low=ohlcv['low'],
            close=ohlcv['close'],
            increasing_line_color='#00C853',
            decreasing_line_color='#FF1744',
            name='Fiyat'
        ))
        
        # EMA çizgileri
        fig.add_trace(go.Scatter(x=ohlcv['timestamp'], y=ohlcv['EMA_20'], 
                                  mode='lines', name='EMA 20', line=dict(color='#2196F3', width=1)))
        fig.add_trace(go.Scatter(x=ohlcv['timestamp'], y=ohlcv['EMA_50'], 
                                  mode='lines', name='EMA 50', line=dict(color='#FF9800', width=1)))
        fig.add_trace(go.Scatter(x=ohlcv['timestamp'], y=ohlcv['EMA_200'], 
                                  mode='lines', name='EMA 200', line=dict(color='#F44336', width=1.5)))
        
        fig.update_layout(
            yaxis_title="Fiyat (USDT)",
            template="plotly_dark",
            height=500,
            margin=dict(l=0, r=0, t=20, b=20),
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # EMA Durumu
        latest = ohlcv.iloc[-1]
        ema_status = []
        if latest['close'] > latest['EMA_20']:
            ema_status.append("🟢 EMA20 üstünde")
        else:
            ema_status.append("🔴 EMA20 altında")
        if latest['close'] > latest['EMA_50']:
            ema_status.append("🟢 EMA50 üstünde")
        else:
            ema_status.append("🔴 EMA50 altında")
        if latest['close'] > latest['EMA_200']:
            ema_status.append("🟢 EMA200 üstünde (Boğa)")
        else:
            ema_status.append("🔴 EMA200 altında (Ayı)")
        
        st.info(" | ".join(ema_status))
    else:
        st.error(f"Grafik yüklenemedi: {ohlcv_err}")


def render_stock_page():
    """Hisse Senedi Sayfası"""
    st.title("📈 Hisse Senedi Analizi")
    
    col1, col2 = st.columns([3, 5])
    with col1:
        stock_symbol = st.text_input("Sembol", value="AAPL", help="THYAO.IS gibi Türk hisseleri için .IS ekleyin")
    
    st.divider()
    
    if stock_symbol.strip():
        data, error = fetch_stock_data(stock_symbol.strip().upper())
        
        if data is not None and not data.empty:
            st.caption(f"📊 {stock_symbol.upper()} - Son 6 Ay")
            
            # Metrikler
            cols = st.columns(4)
            last = data['Close'].iloc[-1]
            prev = data['Close'].iloc[-2] if len(data) > 1 else last
            change = ((last - prev) / prev) * 100
            
            with cols[0]:
                st.metric("💰 Son Fiyat", f"${last:,.2f}", f"{change:+.2f}%")
            with cols[1]:
                st.metric("📈 6Ay Yüksek", f"${data['High'].max():,.2f}")
            with cols[2]:
                st.metric("📉 6Ay Düşük", f"${data['Low'].min():,.2f}")
            with cols[3]:
                st.metric("📊 Ort. Hacim", f"{data['Volume'].mean()/1e6:,.1f}M")
            
            st.divider()
            
            # Grafik
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Kapanış', line=dict(color='#4CAF50', width=2)))
            fig.add_trace(go.Scatter(x=data.index, y=data['EMA_20'], mode='lines', name='EMA 20', line=dict(color='#2196F3', width=1)))
            fig.add_trace(go.Scatter(x=data.index, y=data['EMA_50'], mode='lines', name='EMA 50', line=dict(color='#FF9800', width=1)))
            
            fig.update_layout(
                yaxis_title="Fiyat ($)",
                template="plotly_dark",
                height=400,
                margin=dict(l=0, r=0, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Veri alınamadı: {error}")


def render_onchain_page():
    """On-Chain Bilanço Sayfası - Buffett Modülü"""
    st.title("🔍 On-Chain Bilanço Analizi")
    st.caption("Warren Buffett tarzı temel analiz - 'Bilanço her şeydir'")
    st.divider()
    
    # Protokol Seçimi
    col1, col2 = st.columns([3, 5])
    with col1:
        selected_protocol = st.selectbox("DeFi Protokolü Seç", list(DEFI_PROTOCOLS.keys()))
    
    protocol_slug = DEFI_PROTOCOLS[selected_protocol]
    
    st.divider()
    
    # Veri çek
    with st.spinner(f"{selected_protocol} verileri yükleniyor..."):
        protocol_data, proto_err = fetch_defillama_protocol(protocol_slug)
        treasury_data, treasury_err = fetch_defillama_treasury(protocol_slug)
    
    if protocol_data:
        # Temel Metrikler - tip kontrolü ile
        try:
            tvl = float(protocol_data.get('tvl', 0) or 0)
        except (TypeError, ValueError):
            tvl = 0.0
        
        try:
            mcap = float(protocol_data.get('mcap', 0) or 0)
        except (TypeError, ValueError):
            mcap = 0.0
        
        cols = st.columns(3)
        
        with cols[0]:
            if tvl > 1e9:
                tvl_str = f"${tvl/1e9:.2f}B"
            elif tvl > 0:
                tvl_str = f"${tvl/1e6:.0f}M"
            else:
                tvl_str = "—"
            st.metric("� TVL (Kilitli Değer)", tvl_str)
        
        with cols[1]:
            if mcap > 1e9:
                mcap_str = f"${mcap/1e9:.2f}B"
            elif mcap > 0:
                mcap_str = f"${mcap/1e6:.0f}M"
            else:
                mcap_str = "—"
            st.metric("💎 Market Cap", mcap_str)
        
        with cols[2]:
            mcap_tvl = (mcap / tvl) if tvl > 0 and mcap > 0 else 0
            color = "🟢" if mcap_tvl < 1 else "🟡" if mcap_tvl < 3 else "🔴"
            st.metric(f"{color} Mcap/TVL Oranı", f"{mcap_tvl:.2f}x" if mcap_tvl > 0 else "—")
        
        # P/S Oranı (Yeni Satır)
        st.divider()
        st.subheader("💰 Gelir Analizi (Price-to-Sales)")
        
        revenue_data, revenue_err = fetch_protocol_revenue(protocol_slug)
        
        cols2 = st.columns(3)
        
        with cols2[0]:
            if revenue_data and revenue_data.get('revenue_30d'):
                try:
                    rev_30d = float(revenue_data['revenue_30d'])
                    st.metric("📈 30 Günlük Gelir", f"${rev_30d/1e6:.2f}M")
                except:
                    st.metric("📈 30 Günlük Gelir", "—")
            else:
                st.metric("📈 30 Günlük Gelir", "—")
        
        with cols2[1]:
            if revenue_data and revenue_data.get('revenue_24h'):
                try:
                    rev_24h = float(revenue_data['revenue_24h'])
                    st.metric("📊 24s Gelir", f"${rev_24h/1e3:.1f}K")
                except:
                    st.metric("📊 24s Gelir", "—")
            else:
                st.metric("📊 24s Gelir", "—")
        
        with cols2[2]:
            # P/S = Mcap / (Monthly Revenue * 12)
            if revenue_data and revenue_data.get('revenue_30d') and mcap > 0:
                try:
                    rev_30d = float(revenue_data['revenue_30d'])
                    if rev_30d > 0:
                        annualized_revenue = rev_30d * 12
                        ps_ratio = mcap / annualized_revenue
                        ps_color = "🟢" if ps_ratio < 20 else "🟡" if ps_ratio < 50 else "🔴"
                        st.metric(f"{ps_color} P/S Oranı", f"{ps_ratio:.1f}x")
                    else:
                        st.metric("📉 P/S Oranı", "—")
                except:
                    st.metric("📉 P/S Oranı", "—")
            else:
                st.metric("📉 P/S Oranı", "—")
        
        st.caption("💡 P/S = Market Cap / (Aylık Gelir × 12). Düşük P/S = Potansiyel ucuz.")
        
        st.divider()
        
        # Buffett Skoru
        st.subheader("🎯 Buffett Finansal Sağlık Skoru")
        
        score, details = calculate_buffett_score(mcap, tvl, treasury_data)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Skor gösterimi
            if score >= 8:
                color = "#00C853"
                verdict = "GÜÇLÜ"
            elif score >= 5:
                color = "#FF9800"
                verdict = "ORTA"
            else:
                color = "#FF1744"
                verdict = "ZAYIF"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, {color}22, {color}44); border-radius: 10px; border: 2px solid {color};">
                <h1 style="color: {color}; margin: 0; font-size: 4rem;">{score}/10</h1>
                <p style="color: {color}; margin: 0; font-size: 1.2rem;">{verdict}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**📋 Analiz Detayları:**")
            for detail in details:
                st.write(detail)
        
        st.divider()
        
        # TVL Trendi
        st.subheader("📈 TVL Geçmişi")
        
        tvl_history = protocol_data.get('tvl', [])
        chain_tvls = protocol_data.get('chainTvls', {})
        
        if chain_tvls:
            # En büyük chain'i bul
            main_chain = max(chain_tvls.keys(), key=lambda x: chain_tvls[x].get('tvl', 0) if isinstance(chain_tvls[x], dict) else 0, default=None)
            
            if main_chain and 'tvl' in chain_tvls.get(main_chain, {}):
                tvl_data = chain_tvls[main_chain].get('tvl', [])
                if tvl_data and isinstance(tvl_data, list):
                    df_tvl = pd.DataFrame(tvl_data)
                    if 'date' in df_tvl.columns and 'totalLiquidityUSD' in df_tvl.columns:
                        df_tvl['date'] = pd.to_datetime(df_tvl['date'], unit='s')
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df_tvl['date'],
                            y=df_tvl['totalLiquidityUSD'],
                            mode='lines',
                            fill='tozeroy',
                            line=dict(color='#4CAF50', width=2),
                            name='TVL'
                        ))
                        
                        fig.update_layout(
                            yaxis_title="TVL ($)",
                            template="plotly_dark",
                            height=300,
                            margin=dict(l=0, r=0, t=20, b=20)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("TVL geçmişi formatı desteklenmiyor.")
                else:
                    st.info("TVL geçmişi bulunamadı.")
            else:
                st.info("Zincir TVL verisi mevcut değil.")
        else:
            st.info("Detaylı TVL verisi bulunamadı.")
        
        # Treasury Bilgisi
        if treasury_data:
            st.divider()
            st.subheader("💰 Hazine (Treasury) Durumu")
            
            treasury_tvl = treasury_data.get('tvl', 0)
            st.metric("Toplam Hazine", f"${treasury_tvl/1e6:.1f}M" if treasury_tvl else "Veri yok")
    else:
        st.error(f"Protokol verisi alınamadı: {proto_err}")
        st.info("💡 DeFiLlama API'sine bağlanırken sorun oluştu. Lütfen tekrar deneyin.")


def render_macro_page():
    """Makro Ekonomi Sayfası - Piyasa Pusulası"""
    st.title("📊 Makro Ekonomi")
    st.caption("Küresel piyasa göstergeleri ve korelasyon analizi")
    st.divider()
    
    # Makro verileri çek
    with st.spinner("Makro veriler yükleniyor..."):
        macro_data = fetch_macro_data()
    
    # Metrikler
    st.subheader("🌍 Küresel Göstergeler")
    cols = st.columns(4)
    
    # DXY
    with cols[0]:
        if macro_data.get('DXY'):
            dxy = macro_data['DXY']
            st.metric(
                "💵 DXY (Dolar)",
                f"{dxy['value']:.2f}",
                f"{dxy['change']:+.2f}%"
            )
        else:
            st.metric("💵 DXY", "—")
    
    # US 10Y
    with cols[1]:
        if macro_data.get('US10Y'):
            bonds = macro_data['US10Y']
            st.metric(
                "📜 ABD 10Y Tahvil",
                f"%{bonds['value']:.2f}",
                f"{bonds['change']:+.2f}%"
            )
        else:
            st.metric("📜 ABD 10Y", "—")
    
    # Gold
    with cols[2]:
        if macro_data.get('Gold'):
            gold = macro_data['Gold']
            st.metric(
                "🥇 Altın",
                f"${gold['value']:,.0f}",
                f"{gold['change']:+.2f}%"
            )
        else:
            st.metric("🥇 Altın", "—")
    
    # VIX
    with cols[3]:
        if macro_data.get('VIX'):
            vix = macro_data['VIX']
            vix_status = "🟢" if vix['value'] < 20 else "🟡" if vix['value'] < 30 else "🔴"
            st.metric(
                f"😱 VIX {vix_status}",
                f"{vix['value']:.1f}",
                f"{vix['change']:+.2f}%"
            )
        else:
            st.metric("😱 VIX", "—")
    
    st.divider()
    
    # Korelasyon Analizi
    st.subheader("📈 DXY - Bitcoin Korelasyonu")
    
    col1, col2 = st.columns([3, 5])
    
    with col1:
        crypto_options = {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Gold": "GC=F"}
        selected_asset = st.selectbox("Karşılaştır", list(crypto_options.keys()))
    
    asset_symbol = crypto_options[selected_asset]
    
    with st.spinner("Korelasyon hesaplanıyor..."):
        corr_data, corr_error = fetch_correlation_data(asset_symbol, 60)
    
    if corr_data:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            corr_value = corr_data['correlation']
            
            if corr_value < -0.3:
                corr_color = "#00C853"
                corr_text = "Negatif (BTC için olumlu)"
            elif corr_value > 0.3:
                corr_color = "#FF1744"
                corr_text = "Pozitif (BTC için olumsuz)"
            else:
                corr_color = "#FF9800"
                corr_text = "Nötr"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, {corr_color}22, {corr_color}44); border-radius: 10px; border: 2px solid {corr_color};">
                <h2 style="color: {corr_color}; margin: 0;">{corr_value:.2f}</h2>
                <p style="color: {corr_color}; margin: 0; font-size: 0.9rem;">{corr_text}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.caption(f"Son {corr_data['days']} gün verisi")
        
        with col2:
            # Dual axis chart
            fig = go.Figure()
            
            dxy_hist = corr_data['dxy_data']
            crypto_hist = corr_data['crypto_data']
            
            fig.add_trace(go.Scatter(
                x=dxy_hist.index,
                y=dxy_hist['Close'],
                name='DXY',
                line=dict(color='#2196F3', width=2),
                yaxis='y'
            ))
            
            fig.add_trace(go.Scatter(
                x=crypto_hist.index,
                y=crypto_hist['Close'],
                name=selected_asset,
                line=dict(color='#FF9800', width=2),
                yaxis='y2'
            ))
            
            fig.update_layout(
                template="plotly_dark",
                height=300,
                margin=dict(l=0, r=0, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                yaxis=dict(title="DXY", side="left"),
                yaxis2=dict(title=selected_asset, side="right", overlaying="y")
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"Korelasyon verisi alınamadı: {corr_error}")
    
    st.divider()
    
    # Piyasa Yorumu
    st.subheader("💡 Piyasa Pusulası")
    
    insights = []
    
    if macro_data.get('DXY'):
        dxy_val = macro_data['DXY']['value']
        if dxy_val > 105:
            insights.append("🔴 **Güçlü Dolar**: Risk iştahı düşük, kripto için baskı.")
        elif dxy_val < 100:
            insights.append("🟢 **Zayıf Dolar**: Risk iştahı yüksek, kripto için olumlu.")
        else:
            insights.append("🟡 **Nötr Dolar**: Piyasa yön arıyor.")
    
    if macro_data.get('VIX'):
        vix_val = macro_data['VIX']['value']
        if vix_val > 30:
            insights.append("🔴 **Yüksek Korku**: Volatilite yüksek, dikkatli olun.")
        elif vix_val < 15:
            insights.append("🟢 **Düşük Korku**: Piyasa sakin, risk alınabilir.")
    
    if macro_data.get('US10Y'):
        bond_val = macro_data['US10Y']['value']
        if bond_val > 4.5:
            insights.append("🔴 **Yüksek Faiz**: Hisse ve kripto için baskı.")
        elif bond_val < 3.5:
            insights.append("🟢 **Düşük Faiz**: Risk varlıkları için olumlu.")
    
    if insights:
        for insight in insights:
            st.write(insight)
    else:
        st.info("Piyasa verileri yüklenemedi.")


def render_settings_page():
    """Ayarlar Sayfası"""
    st.title("⚙️ Ayarlar")
    st.divider()
    
    st.subheader("📊 Veri Önbellek Süreleri")
    st.info("""
    - **Kripto Verileri**: 10 dakika
    - **Hisse Verileri**: 15 dakika
    - **On-Chain Verileri**: 10 dakika
    - **Ethereum Ağ**: 1 dakika
    """)
    
    st.divider()
    
    st.subheader("🔗 Veri Kaynakları")
    st.write("- **Kripto**: KuCoin, Kraken (ccxt)")
    st.write("- **Hisse**: Yahoo Finance (yfinance)")
    st.write("- **On-Chain**: DeFiLlama API")
    st.write("- **Ethereum**: Cloudflare, Ankr RPC")
    
    st.divider()
    
    st.subheader("ℹ️ Hakkında")
    st.caption("Finans Terminali v3.0 - Buffett Edition")
    st.caption("Bu uygulama yalnızca bilgilendirme amaçlıdır, yatırım tavsiyesi değildir.")
    
    if st.button("🔄 Önbelleği Temizle"):
        st.cache_data.clear()
        st.success("Önbellek temizlendi!")


# ==================== SIDEBAR NAVİGASYON ====================

def render_sidebar():
    """Sidebar navigasyon"""
    st.sidebar.title("📊 Finans Terminali")
    st.sidebar.caption("Buffett Edition v3.0")
    st.sidebar.divider()
    
    pages = [
        '🏠 Dashboard',
        '🪙 Kripto Analiz',
        '📈 Hisse Senedi',
        '🔍 On-Chain Bilanço',
        '📊 Makro Ekonomi',
        '⚙️ Ayarlar'
    ]
    
    selected = st.sidebar.radio("Sayfa Seçin", pages, label_visibility="collapsed")
    
    st.sidebar.divider()
    st.sidebar.caption("💡 Tüm veriler önbelleğe alınır")
    st.sidebar.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')}")
    
    return selected


# ==================== ANA ROUTER ====================

def main():
    """Ana uygulama"""
    selected_page = render_sidebar()
    
    if selected_page == '🏠 Dashboard':
        render_dashboard()
    elif selected_page == '🪙 Kripto Analiz':
        render_crypto_page()
    elif selected_page == '📈 Hisse Senedi':
        render_stock_page()
    elif selected_page == '🔍 On-Chain Bilanço':
        render_onchain_page()
    elif selected_page == '📊 Makro Ekonomi':
        render_macro_page()
    elif selected_page == '⚙️ Ayarlar':
        render_settings_page()
    
    # Footer
    st.divider()
    st.caption("📊 Finans Terminali | Veriler bilgilendirme amaçlıdır, yatırım tavsiyesi değildir.")


if __name__ == "__main__":
    main()
