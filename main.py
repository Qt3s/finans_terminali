"""
Profesyonel Finans Terminali
Streamlit Cloud için optimize edilmiş, mobil uyumlu dashboard
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Sayfa Konfigürasyonu
st.set_page_config(
    page_title="Finans Terminali",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"  # Mobilde kapalı başlat
)

# Custom CSS - Mobil ve masaüstü uyumu için
st.markdown("""
<style>
    /* Ana konteyner padding ayarı */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Metrik kartları için stil */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
    
    /* Mobil için daha iyi responsive */
    @media (max-width: 768px) {
        [data-testid="stMetricValue"] {
            font-size: 1.5rem;
        }
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)


# ==================== CACHING FONKSİYONLARI ====================

# Desteklenen borsalar ve parite dönüşümleri
EXCHANGE_CONFIGS = [
    {
        'name': 'kucoin',
        'class': 'kucoin',
        'options': {'enableRateLimit': True},
        'symbol_map': {}  # Direkt kullan: BTC/USDT
    },
    {
        'name': 'kraken',
        'class': 'kraken',
        'options': {'enableRateLimit': True},
        'symbol_map': {
            'BTC/USDT': 'BTC/USDT',
            'ETH/USDT': 'ETH/USDT',
            'SOL/USDT': 'SOL/USDT',
            'XRP/USDT': 'XRP/USDT',
            'ADA/USDT': 'ADA/USDT',
            'DOGE/USDT': 'DOGE/USDT',
            'BNB/USDT': 'BNB/USDT',  # Kraken'de olmayabilir, fallback
        }
    },
]


def get_exchange_instance(config):
    """Borsa instance'ı oluşturur."""
    import ccxt
    exchange_class = getattr(ccxt, config['class'])
    return exchange_class(config['options'])


@st.cache_data(ttl=300, show_spinner=False)  # 5 dakika cache
def fetch_crypto_ohlcv(symbol: str, timeframe: str, limit: int = 100):
    """
    Birden fazla borsadan OHLCV verisi çeker (fallback mekanizması).
    KuCoin -> Kraken sırasıyla dener.
    """
    import ccxt
    errors = []
    
    for config in EXCHANGE_CONFIGS:
        try:
            exchange = get_exchange_instance(config)
            
            # Sembol dönüşümü (gerekirse)
            mapped_symbol = config['symbol_map'].get(symbol, symbol)
            
            ohlcv = exchange.fetch_ohlcv(mapped_symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df, None, config['name']  # Başarılı borsa adını da döndür
            
        except Exception as e:
            errors.append(f"{config['name']}: {str(e)}")
            continue
    
    # Tüm borsalar başarısız olduysa
    return None, " | ".join(errors), None


@st.cache_data(ttl=300, show_spinner=False)  # 5 dakika cache
def fetch_crypto_ticker(symbol: str):
    """
    Birden fazla borsadan anlık fiyat bilgisi çeker (fallback mekanizması).
    KuCoin -> Kraken sırasıyla dener.
    """
    import ccxt
    errors = []
    
    for config in EXCHANGE_CONFIGS:
        try:
            exchange = get_exchange_instance(config)
            
            # Sembol dönüşümü (gerekirse)
            mapped_symbol = config['symbol_map'].get(symbol, symbol)
            
            ticker = exchange.fetch_ticker(mapped_symbol)
            return ticker, None, config['name']  # Başarılı borsa adını da döndür
            
        except Exception as e:
            errors.append(f"{config['name']}: {str(e)}")
            continue
    
    # Tüm borsalar başarısız olduysa
    return None, " | ".join(errors), None


@st.cache_data(ttl=900, show_spinner=False)  # 15 dakika cache (rate limit için artırıldı)
def fetch_stock_data(symbol: str, period: str = "6mo"):
    """
    Yahoo Finance'den hisse senedi verisi çeker.
    Retry mekanizması ile rate limiting'e karşı koruma.
    """
    import time
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                return None, f"'{symbol}' sembolü için veri bulunamadı."
            
            return hist, None
        except Exception as e:
            error_msg = str(e).lower()
            if "rate" in error_msg or "too many" in error_msg:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff: 1, 2, 4 saniye
                    continue
            return None, str(e)
    
    return None, "Rate limit aşıldı. Lütfen birkaç dakika bekleyin."


@st.cache_data(ttl=60, show_spinner=False)  # 1 dakika cache (on-chain daha dinamik)
def fetch_ethereum_data():
    """
    Ethereum ağından blok ve gas bilgisi çeker.
    Ücretsiz genel RPC endpoint kullanır.
    """
    try:
        from web3 import Web3
        
        # Ücretsiz genel Ethereum RPC noktaları
        rpc_endpoints = [
            "https://cloudflare-eth.com",
            "https://eth.llamarpc.com",
            "https://rpc.ankr.com/eth",
        ]
        
        for rpc_url in rpc_endpoints:
            try:
                w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={'timeout': 10}))
                if w3.is_connected():
                    block_number = w3.eth.block_number
                    gas_price_wei = w3.eth.gas_price
                    gas_price_gwei = round(gas_price_wei / 1e9, 2)
                    
                    return {
                        'block_number': block_number,
                        'gas_price_gwei': gas_price_gwei,
                        'rpc_used': rpc_url
                    }, None
            except:
                continue
        
        return None, "Tüm Ethereum RPC noktalarına bağlanılamadı."
    except Exception as e:
        return None, str(e)


# ==================== SIDEBAR (YAN MENÜ) ====================

st.sidebar.title("⚙️ Ayarlar")

# Kripto Ayarları
st.sidebar.header("🪙 Kripto")
crypto_symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT", "DOGE/USDT"]
selected_crypto = st.sidebar.selectbox("Parite Seç", crypto_symbols, index=0)
timeframes = {"1 Saat": "1h", "4 Saat": "4h", "1 Gün": "1d", "1 Hafta": "1w"}
selected_timeframe_label = st.sidebar.selectbox("Zaman Dilimi", list(timeframes.keys()), index=1)
selected_timeframe = timeframes[selected_timeframe_label]

st.sidebar.divider()

# Hisse Senedi Ayarları
st.sidebar.header("📈 Hisse Senedi")
stock_symbol = st.sidebar.text_input(
    "Sembol Gir", 
    value="AAPL",
    help="Örnek: AAPL, GOOGL, MSFT, THYAO.IS (Türk hisseleri için .IS ekleyin)"
)

st.sidebar.divider()

# Bilgi
st.sidebar.info("💡 Veriler her 5 dakikada bir güncellenir. On-chain verileri 1 dakikada bir yenilenir.")


# ==================== ANA EKRAN ====================

st.title("📊 Finans Terminali")

# Sekmeler
tab_crypto, tab_stock, tab_onchain = st.tabs(["🪙 Kripto", "📈 Hisse Senedi", "⛓️ On-Chain"])


# ==================== SEKME 1: KRİPTO ====================

with tab_crypto:
    st.subheader(f"{selected_crypto} - {selected_timeframe_label}")
    
    # Anlık fiyat bilgisi
    with st.spinner("Fiyat bilgisi alınıyor..."):
        ticker_data, ticker_error, ticker_exchange = fetch_crypto_ticker(selected_crypto)
    
    if ticker_error:
        st.error(f"⚠️ Fiyat verisi alınamadı: {ticker_error}")
    elif ticker_data:
        # Hangi borsadan geldiğini göster
        st.caption(f"📡 Veri kaynağı: **{ticker_exchange.upper()}**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_price = ticker_data.get('last', 0)
            change_percent = ticker_data.get('percentage', 0)
            st.metric(
                label="Anlık Fiyat",
                value=f"${current_price:,.2f}",
                delta=f"{change_percent:+.2f}%"
            )
        
        with col2:
            high_24h = ticker_data.get('high', 0)
            st.metric(label="24s Yüksek", value=f"${high_24h:,.2f}")
        
        with col3:
            low_24h = ticker_data.get('low', 0)
            st.metric(label="24s Düşük", value=f"${low_24h:,.2f}")
    
    st.divider()
    
    # OHLCV Verisi ve Mum Grafiği
    with st.spinner("Grafik verisi yükleniyor..."):
        ohlcv_data, ohlcv_error, ohlcv_exchange = fetch_crypto_ohlcv(selected_crypto, selected_timeframe)
    
    if ohlcv_error:
        st.error(f"⚠️ Grafik verisi alınamadı: {ohlcv_error}")
        st.warning("Lütfen birkaç dakika bekleyip tekrar deneyin veya başka bir parite seçin.")
    elif ohlcv_data is not None and not ohlcv_data.empty:
        # Plotly Candlestick Grafiği
        fig = go.Figure(data=[go.Candlestick(
            x=ohlcv_data['timestamp'],
            open=ohlcv_data['open'],
            high=ohlcv_data['high'],
            low=ohlcv_data['low'],
            close=ohlcv_data['close'],
            increasing_line_color='#00C853',  # Yeşil
            decreasing_line_color='#FF1744',  # Kırmızı
            name=selected_crypto
        )])
        
        fig.update_layout(
            title=None,
            yaxis_title="Fiyat (USDT)",
            xaxis_title=None,
            template="plotly_dark",
            height=500,
            margin=dict(l=0, r=0, t=20, b=20),
            xaxis_rangeslider_visible=False,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Hacim bilgisi
        total_volume = ohlcv_data['volume'].sum()
        st.caption(f"📊 Toplam İşlem Hacmi (son {len(ohlcv_data)} mum): {total_volume:,.0f}")
    else:
        st.warning("Grafik verisi boş döndü. Lütfen başka bir parite veya zaman dilimi deneyin.")


# ==================== SEKME 2: HİSSE SENEDİ ====================

with tab_stock:
    st.subheader(f"📈 {stock_symbol.upper()} - Son 6 Ay")
    
    if stock_symbol.strip():
        with st.spinner("Hisse verisi alınıyor..."):
            stock_data, stock_error = fetch_stock_data(stock_symbol.strip().upper())
        
        if stock_error:
            st.error(f"⚠️ Hisse verisi alınamadı: {stock_error}")
            st.info("💡 İpucu: Türk hisseleri için '.IS' eki kullanın (örn: THYAO.IS)")
        elif stock_data is not None and not stock_data.empty:
            # Metrikleri göster
            col1, col2, col3 = st.columns(3)
            
            with col1:
                last_close = stock_data['Close'].iloc[-1]
                prev_close = stock_data['Close'].iloc[-2] if len(stock_data) > 1 else last_close
                change = ((last_close - prev_close) / prev_close) * 100
                st.metric(
                    label="Son Kapanış",
                    value=f"${last_close:,.2f}",
                    delta=f"{change:+.2f}%"
                )
            
            with col2:
                high_6m = stock_data['High'].max()
                st.metric(label="6 Ay Yüksek", value=f"${high_6m:,.2f}")
            
            with col3:
                low_6m = stock_data['Low'].min()
                st.metric(label="6 Ay Düşük", value=f"${low_6m:,.2f}")
            
            st.divider()
            
            # Çizgi grafiği
            st.line_chart(stock_data['Close'], use_container_width=True)
            
            st.caption(f"📅 Veri aralığı: {stock_data.index[0].strftime('%d/%m/%Y')} - {stock_data.index[-1].strftime('%d/%m/%Y')}")
        else:
            st.warning("Hisse verisi bulunamadı.")
    else:
        st.info("👈 Yan menüden bir hisse sembolü girin.")


# ==================== SEKME 3: ON-CHAIN ====================

with tab_onchain:
    st.subheader("⛓️ Ethereum Ağ Durumu")
    
    with st.spinner("Ethereum ağına bağlanılıyor..."):
        eth_data, eth_error = fetch_ethereum_data()
    
    if eth_error:
        st.error(f"⚠️ Ethereum verisi alınamadı: {eth_error}")
        st.warning("Lütfen birkaç dakika bekleyip tekrar deneyin. RPC noktaları geçici olarak yanıt vermiyor olabilir.")
    elif eth_data:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="📦 Son Blok Numarası",
                value=f"{eth_data['block_number']:,}"
            )
        
        with col2:
            gas_gwei = eth_data['gas_price_gwei']
            # Gas seviyesi göstergesi
            if gas_gwei < 20:
                gas_status = "🟢 Düşük"
            elif gas_gwei < 50:
                gas_status = "🟡 Orta"
            else:
                gas_status = "🔴 Yüksek"
            
            st.metric(
                label=f"⛽ Gas Ücreti ({gas_status})",
                value=f"{gas_gwei} Gwei"
            )
        
        st.divider()
        
        # Ek bilgi
        st.info(f"""
        **ℹ️ Ethereum Ağ Bilgisi**
        
        - **RPC Endpoint:** {eth_data['rpc_used']}
        - **Gas Öneri:** {"İşlem yapmak için uygun zaman!" if gas_gwei < 30 else "Gas ücretleri yüksek, bekleyebilirsiniz."}
        
        *Veriler her dakika güncellenir.*
        """)
    else:
        st.warning("Ethereum ağ verisi alınamadı.")


# ==================== FOOTER ====================

st.divider()
st.caption("📊 Finans Terminali | Veriler yalnızca bilgilendirme amaçlıdır, yatırım tavsiyesi değildir.")
st.caption(f"🕐 Son güncelleme: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
