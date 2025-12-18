"""
Profesyonel Finans Terminali v2.0
Modüler mimari, sidebar navigasyon, dinamik filtreler
Streamlit Cloud için optimize edilmiş, mobil uyumlu dashboard
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
# ==================== SAYFA KONFİGÜRASYONU ====================
st.set_page_config(
    page_title="Finans Terminali",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom CSS - Mobil ve masaüstü uyumu
st.markdown("""
<style>
    /* Ana konteyner padding ayarı */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Metrik kartları için stil */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: bold;
    }
    
    /* Sidebar başlık stili */
    [data-testid="stSidebar"] h1 {
        font-size: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #4CAF50;
    }
    
    /* Mobil için responsive */
    @media (max-width: 768px) {
        [data-testid="stMetricValue"] {
            font-size: 1.3rem;
        }
        .main .block-container {
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }
    }
    
    /* Container kartları için stil */
    .stContainer {
        border-radius: 10px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)
# ==================== BORSA KONFİGÜRASYONU ====================
EXCHANGE_CONFIGS = [
    {
        'name': 'kucoin',
        'class': 'kucoin',
        'options': {'enableRateLimit': True},
        'symbol_map': {}
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
            'BNB/USDT': 'BNB/USDT',
        }
    },
]
CRYPTO_SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT", "DOGE/USDT"]
TIMEFRAMES = {"1 Saat": "1h", "4 Saat": "4h", "1 Gün": "1d", "1 Hafta": "1w"}
# ==================== VERİ ÇEKİCİ FONKSİYONLAR ====================
def get_exchange_instance(config):
    """Borsa instance'ı oluşturur."""
    import ccxt
    exchange_class = getattr(ccxt, config['class'])
    return exchange_class(config['options'])
@st.cache_data(ttl=300, show_spinner=False)
def fetch_crypto_ticker(symbol: str):
    """Birden fazla borsadan anlık fiyat bilgisi çeker (fallback)."""
    import ccxt
    errors = []
    
    for config in EXCHANGE_CONFIGS:
        try:
            exchange = get_exchange_instance(config)
            mapped_symbol = config['symbol_map'].get(symbol, symbol)
            ticker = exchange.fetch_ticker(mapped_symbol)
            return ticker, None, config['name']
        except Exception as e:
            errors.append(f"{config['name']}: {str(e)}")
            continue
    
    return None, " | ".join(errors), None
@st.cache_data(ttl=300, show_spinner=False)
def fetch_crypto_ohlcv(symbol: str, timeframe: str, limit: int = 100):
    """Birden fazla borsadan OHLCV verisi çeker (fallback)."""
    import ccxt
    errors = []
    
    for config in EXCHANGE_CONFIGS:
        try:
            exchange = get_exchange_instance(config)
            mapped_symbol = config['symbol_map'].get(symbol, symbol)
            ohlcv = exchange.fetch_ohlcv(mapped_symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df, None, config['name']
        except Exception as e:
            errors.append(f"{config['name']}: {str(e)}")
            continue
    
    return None, " | ".join(errors), None
@st.cache_data(ttl=900, show_spinner=False)
def fetch_stock_data(symbol: str, period: str = "6mo"):
    """Yahoo Finance'den hisse senedi verisi çeker."""
    import time
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                return None, f"'{symbol}' için veri bulunamadı."
            
            return hist, None
        except Exception as e:
            error_msg = str(e).lower()
            if "rate" in error_msg or "too many" in error_msg:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
            return None, str(e)
    
    return None, "Rate limit aşıldı. Lütfen bekleyin."
@st.cache_data(ttl=60, show_spinner=False)
def fetch_ethereum_data():
    """Ethereum ağından blok ve gas bilgisi çeker."""
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
        
        return None, "Tüm RPC noktalarına bağlanılamadı."
    except Exception as e:
        return None, str(e)
# ==================== SAYFA FONKSİYONLARI ====================
def show_dashboard():
    """Ana Dashboard - Piyasa Özeti"""
    st.title("🏠 Piyasa Özeti")
    st.caption("Anlık piyasa durumu ve önemli varlıklar")
    
    st.divider()
    
    # Kripto Özet Bölümü
    st.subheader("🪙 Kripto Piyasası")
    
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        # Bitcoin
        with col1:
            with st.spinner("BTC..."):
                btc_data, btc_error, _ = fetch_crypto_ticker("BTC/USDT")
            if btc_data:
                st.metric(
                    label="Bitcoin (BTC)",
                    value=f"${btc_data.get('last', 0):,.0f}",
                    delta=f"{btc_data.get('percentage', 0):+.2f}%"
                )
            else:
                st.metric(label="Bitcoin (BTC)", value="—", delta="Veri yok")
        
        # Ethereum
        with col2:
            with st.spinner("ETH..."):
                eth_data, eth_error, _ = fetch_crypto_ticker("ETH/USDT")
            if eth_data:
                st.metric(
                    label="Ethereum (ETH)",
                    value=f"${eth_data.get('last', 0):,.0f}",
                    delta=f"{eth_data.get('percentage', 0):+.2f}%"
                )
            else:
                st.metric(label="Ethereum (ETH)", value="—", delta="Veri yok")
        
        # Solana
        with col3:
            with st.spinner("SOL..."):
                sol_data, sol_error, _ = fetch_crypto_ticker("SOL/USDT")
            if sol_data:
                st.metric(
                    label="Solana (SOL)",
                    value=f"${sol_data.get('last', 0):,.2f}",
                    delta=f"{sol_data.get('percentage', 0):+.2f}%"
                )
            else:
                st.metric(label="Solana (SOL)", value="—", delta="Veri yok")
    
    st.divider()
    
    # Hisse Senedi Özet Bölümü
    st.subheader("📈 Hisse Senedi Piyasası")
    
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        stock_list = [
            ("AAPL", "Apple"),
            ("GOOGL", "Google"),
            ("MSFT", "Microsoft")
        ]
        
        for col, (symbol, name) in zip([col1, col2, col3], stock_list):
            with col:
                with st.spinner(f"{symbol}..."):
                    stock_data, stock_error = fetch_stock_data(symbol, "5d")
                if stock_data is not None and not stock_data.empty:
                    last_close = stock_data['Close'].iloc[-1]
                    prev_close = stock_data['Close'].iloc[-2] if len(stock_data) > 1 else last_close
                    change = ((last_close - prev_close) / prev_close) * 100
                    st.metric(
                        label=f"{name} ({symbol})",
                        value=f"${last_close:,.2f}",
                        delta=f"{change:+.2f}%"
                    )
                else:
                    st.metric(label=f"{name} ({symbol})", value="—", delta="Veri yok")
    
    st.divider()
    
    # Ethereum Ağ Durumu
    st.subheader("⛓️ Ethereum Ağ Durumu")
    
    with st.container():
        col1, col2 = st.columns(2)
        
        with st.spinner("Ethereum ağına bağlanılıyor..."):
            eth_chain, eth_error = fetch_ethereum_data()
        
        if eth_chain:
            with col1:
                st.metric(
                    label="📦 Son Blok",
                    value=f"{eth_chain['block_number']:,}"
                )
            with col2:
                gas_gwei = eth_chain['gas_price_gwei']
                gas_status = "🟢" if gas_gwei < 20 else "🟡" if gas_gwei < 50 else "🔴"
                st.metric(
                    label=f"⛽ Gas Ücreti {gas_status}",
                    value=f"{gas_gwei} Gwei"
                )
        else:
            st.warning("Ethereum ağ verisi alınamadı.")
def show_crypto_page():
    """Kripto Terminal Sayfası"""
    st.title("🪙 Kripto Terminal")
    
    # Sayfa içi filtreler
    col_filter1, col_filter2, col_spacer = st.columns([2, 2, 4])
    
    with col_filter1:
        selected_crypto = st.selectbox(
            "Parite Seç",
            CRYPTO_SYMBOLS,
            index=CRYPTO_SYMBOLS.index(st.session_state.get('crypto_symbol', 'BTC/USDT')) if st.session_state.get('crypto_symbol', 'BTC/USDT') in CRYPTO_SYMBOLS else 0,
            key='crypto_symbol_select'
        )
        st.session_state['crypto_symbol'] = selected_crypto
    
    with col_filter2:
        tf_list = list(TIMEFRAMES.keys())
        selected_tf_label = st.selectbox(
            "Zaman Dilimi",
            tf_list,
            index=tf_list.index(st.session_state.get('crypto_timeframe', '4 Saat')) if st.session_state.get('crypto_timeframe', '4 Saat') in tf_list else 1,
            key='crypto_tf_select'
        )
        st.session_state['crypto_timeframe'] = selected_tf_label
    
    selected_timeframe = TIMEFRAMES.get(selected_tf_label, '4h')
    
    st.divider()
    
    # Anlık Fiyat Bilgisi
    with st.container():
        with st.spinner("Fiyat bilgisi alınıyor..."):
            ticker_data, ticker_error, exchange_name = fetch_crypto_ticker(selected_crypto)
        
        if ticker_error:
            st.error(f"⚠️ Fiyat verisi alınamadı: {ticker_error}")
        elif ticker_data:
            st.caption(f"Veri kaynağı: **{exchange_name.upper()}**")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="💰 Anlık Fiyat",
                    value=f"${ticker_data.get('last', 0):,.2f}",
                    delta=f"{ticker_data.get('percentage', 0):+.2f}%"
                )
            
            with col2:
                st.metric(label="📈 24s Yüksek", value=f"${ticker_data.get('high', 0):,.2f}")
            
            with col3:
                st.metric(label="📉 24s Düşük", value=f"${ticker_data.get('low', 0):,.2f}")
            
            with col4:
                volume = ticker_data.get('quoteVolume', 0) or 0
                st.metric(label="📊 24s Hacim", value=f"${volume/1e6:,.1f}M")
    
    st.divider()
    
    # Mum Grafiği
    with st.container():
        st.subheader("📊 Fiyat Grafiği")
        
        with st.spinner("Grafik yükleniyor..."):
            ohlcv_data, ohlcv_error, ohlcv_exchange = fetch_crypto_ohlcv(selected_crypto, selected_timeframe)
        
        if ohlcv_error:
            st.error(f"⚠️ Grafik verisi alınamadı: {ohlcv_error}")
        elif ohlcv_data is not None and not ohlcv_data.empty:
            fig = go.Figure(data=[go.Candlestick(
                x=ohlcv_data['timestamp'],
                open=ohlcv_data['open'],
                high=ohlcv_data['high'],
                low=ohlcv_data['low'],
                close=ohlcv_data['close'],
                increasing_line_color='#00C853',
                decreasing_line_color='#FF1744',
                name=selected_crypto
            )])
            
            fig.update_layout(
                yaxis_title="Fiyat (USDT)",
                template="plotly_dark",
                height=500,
                margin=dict(l=0, r=0, t=20, b=20),
                xaxis_rangeslider_visible=False,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            total_volume = ohlcv_data['volume'].sum()
            st.caption(f"📊 Toplam Hacim (son {len(ohlcv_data)} mum): {total_volume:,.0f}")
        else:
            st.warning("Grafik verisi yüklenemedi.")
def show_stock_page():
    """Hisse Senedi Sayfası"""
    st.title("📈 Hisse Senedi Terminali")
    
    # Sayfa içi filtre
    col_filter, col_spacer = st.columns([3, 5])
    
    with col_filter:
        stock_symbol = st.text_input(
            "Hisse Sembolü Gir",
            value=st.session_state.get('stock_symbol', 'AAPL'),
            help="Örnek: AAPL, GOOGL, MSFT, THYAO.IS (Türk hisseleri için .IS ekleyin)",
            key='stock_symbol_input'
        )
        st.session_state['stock_symbol'] = stock_symbol
    
    st.divider()
    
    if stock_symbol.strip():
        st.caption(f"📊 {stock_symbol.upper()} - Son 6 Ay")
        with st.container():
            with st.spinner("Hisse verisi alınıyor..."):
                stock_data, stock_error = fetch_stock_data(stock_symbol.strip().upper())
            
            if stock_error:
                st.error(f"⚠️ Hisse verisi alınamadı: {stock_error}")
                st.info("💡 Türk hisseleri için '.IS' eki kullanın (örn: THYAO.IS)")
            elif stock_data is not None and not stock_data.empty:
                # Metrikler
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    last_close = stock_data['Close'].iloc[-1]
                    prev_close = stock_data['Close'].iloc[-2] if len(stock_data) > 1 else last_close
                    change = ((last_close - prev_close) / prev_close) * 100
                    st.metric(
                        label="💰 Son Kapanış",
                        value=f"${last_close:,.2f}",
                        delta=f"{change:+.2f}%"
                    )
                
                with col2:
                    st.metric(label="📈 6 Ay Yüksek", value=f"${stock_data['High'].max():,.2f}")
                
                with col3:
                    st.metric(label="📉 6 Ay Düşük", value=f"${stock_data['Low'].min():,.2f}")
                
                with col4:
                    avg_volume = stock_data['Volume'].mean()
                    st.metric(label="📊 Ort. Hacim", value=f"{avg_volume/1e6:,.1f}M")
                
                st.divider()
                
                # Grafik
                st.subheader("📊 Fiyat Grafiği")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=stock_data['Close'],
                    mode='lines',
                    name='Kapanış',
                    line=dict(color='#4CAF50', width=2)
                ))
                
                fig.update_layout(
                    yaxis_title="Fiyat ($)",
                    template="plotly_dark",
                    height=400,
                    margin=dict(l=0, r=0, t=20, b=20),
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.caption(f"📅 Veri: {stock_data.index[0].strftime('%d/%m/%Y')} - {stock_data.index[-1].strftime('%d/%m/%Y')}")
            else:
                st.warning("Hisse verisi bulunamadı.")
    else:
        st.info("☝️ Yukarıdan bir hisse sembolü girin.")
def show_onchain_page():
    """On-Chain Analiz Sayfası"""
    st.title("🔗 On-Chain Analiz")
    st.caption("Ethereum ağı verileri ve metrikleri")
    st.divider()
    
    with st.container():
        st.subheader("⛓️ Ethereum Ağ Durumu")
        
        with st.spinner("Ethereum ağına bağlanılıyor..."):
            eth_data, eth_error = fetch_ethereum_data()
        
        if eth_error:
            st.error(f"⚠️ Ethereum verisi alınamadı: {eth_error}")
        elif eth_data:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="📦 Son Blok Numarası",
                    value=f"{eth_data['block_number']:,}"
                )
            
            with col2:
                gas_gwei = eth_data['gas_price_gwei']
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
            
            # Bilgi kutusu
            st.info(f"""
            **ℹ️ Ethereum Ağ Bilgisi**
            
            - **RPC Endpoint:** {eth_data['rpc_used']}
            - **Gas Öneri:** {"İşlem yapmak için uygun zaman!" if gas_gwei < 30 else "Gas ücretleri yüksek, bekleyebilirsiniz."}
            
            *Veriler her dakika güncellenir.*
            """)
        else:
            st.warning("Ethereum ağ verisi alınamadı.")
    
    # Gelecek özellikler için placeholder
    st.divider()
    st.subheader("🔮 Yakında Eklenecek")
    st.caption("• Whale Tracker  • DeFi TVL  • NFT Floor Prices")
# ==================== SIDEBAR NAVİGASYON ====================
def render_sidebar():
    """Sidebar - sadece navigasyon"""
    
    st.sidebar.title("📊 Finans Terminali")
    st.sidebar.divider()
    
    # Ana Navigasyon
    pages = ['🏠 Dashboard', '🪙 Kripto Terminal', '📈 Hisse Senedi', '🔗 On-Chain Analiz']
    selected_page = st.sidebar.radio("Sayfa Seçin", pages, index=0, label_visibility="collapsed")
    
    # Footer
    st.sidebar.divider()
    st.sidebar.caption("💡 Veriler cache'lenir.")
    st.sidebar.caption("Kripto: 5dk | Hisse: 15dk | On-chain: 1dk")
    st.sidebar.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')}")
    
    return selected_page
# ==================== ANA ROUTER ====================
def main():
    """Ana uygulama router'ı"""
    
    # Sidebar render et ve sayfa seçimini al
    selected_page = render_sidebar()
    
    # Seçilen sayfayı göster
    if selected_page == '🏠 Dashboard':
        show_dashboard()
    elif selected_page == '🪙 Kripto Terminal':
        show_crypto_page()
    elif selected_page == '📈 Hisse Senedi':
        show_stock_page()
    elif selected_page == '🔗 On-Chain Analiz':
        show_onchain_page()
    
    # Footer
    st.divider()
    st.caption("📊 Finans Terminali v2.0 | Veriler yalnızca bilgilendirme amaçlıdır.")
# Uygulamayı başlat
if __name__ == "__main__":
    main()
