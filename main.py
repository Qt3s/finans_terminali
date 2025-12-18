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


@st.cache_data(ttl=21600, show_spinner=False)  # 6 saat cache
def fetch_macro_data():
    """Genişletilmiş makro ekonomi verileri."""
    import yfinance as yf
    
    symbols = {
        'DXY': 'DX-Y.NYB',      # Dolar Endeksi
        'US10Y': '^TNX',         # ABD 10Y Tahvil
        'US02Y': '^IRX',         # ABD 2Y (yaklaşık - 13 hafta)
        'VIX': '^VIX',           # Korku Endeksi
        'Gold': 'GC=F',          # Altın
        'Silver': 'SI=F',        # Gümüş
        'Oil': 'CL=F',           # WTI Petrol
        'USDJPY': 'JPY=X',       # USD/JPY (Carry Trade)
        'TLT': 'TLT',            # Uzun vadeli tahvil ETF (likidite proxy)
    }
    
    results = {}
    
    for name, symbol in symbols.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='60d')
            
            if not hist.empty:
                # Float32 optimizasyonu
                last = float(hist['Close'].iloc[-1])
                prev = float(hist['Close'].iloc[-2]) if len(hist) > 1 else last
                change = ((last - prev) / prev) * 100 if prev != 0 else 0
                
                # 5 günlük değişim
                prev_5d = float(hist['Close'].iloc[-5]) if len(hist) >= 5 else float(hist['Close'].iloc[0])
                change_5d = ((last - prev_5d) / prev_5d) * 100 if prev_5d != 0 else 0
                
                # 30 günlük değişim
                prev_30d = float(hist['Close'].iloc[0]) if len(hist) >= 20 else float(hist['Close'].iloc[0])
                change_30d = ((last - prev_30d) / prev_30d) * 100 if prev_30d != 0 else 0
                
                results[name] = {
                    'value': last,
                    'change': change,
                    'change_5d': change_5d,
                    'change_30d': change_30d,
                    'history': hist[['Close']].astype('float32')  # Sadece Close, float32
                }
            else:
                results[name] = None
        except Exception as e:
            results[name] = None
    
    return results


@st.cache_data(ttl=21600, show_spinner=False)  # 6 saat cache
def fetch_yield_curve_data():
    """Getiri eğrisi verisi (10Y-2Y spread)."""
    import yfinance as yf
    
    try:
        # 10 Yıllık ve 2 Yıllık tahvil getirisi
        us10y = yf.Ticker('^TNX')
        us02y = yf.Ticker('^IRX')  # 13 hafta T-Bill (2Y proxy)
        
        hist_10y = us10y.history(period='1y')
        hist_02y = us02y.history(period='1y')
        
        if hist_10y.empty or hist_02y.empty:
            return None, "Tahvil verisi alınamadı"
        
        # Son değerler
        y10_last = float(hist_10y['Close'].iloc[-1])
        y02_last = float(hist_02y['Close'].iloc[-1])
        
        # Spread (10Y - 2Y)
        spread = y10_last - y02_last
        
        # Tarihsel spread hesapla
        hist_10y.index = hist_10y.index.date
        hist_02y.index = hist_02y.index.date
        
        # Ortak tarihleri bul
        common_dates = set(hist_10y.index) & set(hist_02y.index)
        
        spread_history = []
        for date in sorted(common_dates):
            try:
                s10 = float(hist_10y.loc[date, 'Close'])
                s02 = float(hist_02y.loc[date, 'Close'])
                spread_history.append({'date': date, 'spread': s10 - s02})
            except:
                continue
        
        return {
            'us10y': y10_last,
            'us02y': y02_last,
            'spread': spread,
            'inverted': spread < 0,
            'history': spread_history[-60:] if spread_history else []  # Son 60 gün
        }, None
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=21600, show_spinner=False)  # 6 saat cache
def fetch_liquidity_proxy():
    """
    Likidite Proxy Endeksi.
    
    Gerçek Fed bilançosu verisi için FRED API key gerekiyor.
    Alternatif olarak TLT (uzun vadeli tahvil ETF) ve M2V kullanıyoruz.
    TLT yükselirse → faizler düşüyor → likidite artıyor
    """
    import yfinance as yf
    
    try:
        # TLT: iShares 20+ Year Treasury Bond ETF
        # Likidite proxy'si olarak kullanılır
        tlt = yf.Ticker('TLT')
        spy = yf.Ticker('SPY')  # S&P 500 ETF
        btc = yf.Ticker('BTC-USD')
        
        tlt_hist = tlt.history(period='1y')
        spy_hist = spy.history(period='1y')
        btc_hist = btc.history(period='1y')
        
        if tlt_hist.empty:
            return None, "TLT verisi alınamadı"
        
        tlt_last = float(tlt_hist['Close'].iloc[-1])
        tlt_prev = float(tlt_hist['Close'].iloc[-30]) if len(tlt_hist) >= 30 else float(tlt_hist['Close'].iloc[0])
        tlt_change = ((tlt_last - tlt_prev) / tlt_prev) * 100
        
        # Likidite skoru: TLT yükseliyorsa likidite artıyor
        if tlt_change > 5:
            liquidity_trend = "ARTIYOR"
            liquidity_score = 20
        elif tlt_change < -5:
            liquidity_trend = "AZALIYOR"
            liquidity_score = -20
        else:
            liquidity_trend = "STABIL"
            liquidity_score = 0
        
        # BTC ve TLT tarihsel karşılaştırma
        btc_history = btc_hist[['Close']].copy() if not btc_hist.empty else None
        tlt_history = tlt_hist[['Close']].copy()
        
        return {
            'tlt_value': tlt_last,
            'tlt_change_30d': tlt_change,
            'liquidity_trend': liquidity_trend,
            'liquidity_score': liquidity_score,
            'tlt_history': tlt_history.astype('float32'),
            'btc_history': btc_history.astype('float32') if btc_history is not None else None
        }, None
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=43200, show_spinner=False)  # 12 saat cache - ağır veri
def fetch_credit_and_liquidity_data():
    """
    Kredi Riski ve Küresel Likidite Verileri.
    
    FRED API key olmadan proxy'ler kullanılır:
    - HYG: iShares High Yield Corporate Bond ETF (kredi spreadi proxy)
    - LQD: Investment Grade Bond ETF
    - TIP: TIPS ETF (reel faiz proxy)
    - HG=F: Copper futures
    - GC=F: Gold futures
    """
    import yfinance as yf
    import numpy as np
    
    try:
        # ETF ve emtia verileri
        hyg = yf.Ticker('HYG')  # High Yield Bond ETF
        lqd = yf.Ticker('LQD')  # Investment Grade Bond ETF
        tip = yf.Ticker('TIP')  # TIPS ETF (reel faiz proxy)
        copper = yf.Ticker('HG=F')  # Bakır
        gold = yf.Ticker('GC=F')  # Altın
        
        hyg_hist = hyg.history(period='6mo')
        lqd_hist = lqd.history(period='6mo')
        tip_hist = tip.history(period='6mo')
        copper_hist = copper.history(period='6mo')
        gold_hist = gold.history(period='6mo')
        
        results = {}
        
        # HY Spread Proxy: HYG/LQD oranı (düşükse spread yüksek = risk yüksek)
        if not hyg_hist.empty and not lqd_hist.empty:
            hyg_last = float(hyg_hist['Close'].iloc[-1])
            lqd_last = float(lqd_hist['Close'].iloc[-1])
            hyg_lqd_ratio = hyg_last / lqd_last
            
            hyg_prev = float(hyg_hist['Close'].iloc[-30]) if len(hyg_hist) >= 30 else hyg_last
            lqd_prev = float(lqd_hist['Close'].iloc[-30]) if len(lqd_hist) >= 30 else lqd_last
            hyg_lqd_prev = hyg_prev / lqd_prev
            
            ratio_change = ((hyg_lqd_ratio - hyg_lqd_prev) / hyg_lqd_prev) * 100
            
            # Oran düşüyorsa = HY kötüleşiyor = kredi riski artıyor
            if ratio_change < -3:
                credit_risk = "YÜKSEK"
                credit_score = -20
            elif ratio_change > 3:
                credit_risk = "DÜŞÜK"
                credit_score = 15
            else:
                credit_risk = "NORMAL"
                credit_score = 0
            
            results['credit'] = {
                'hyg_lqd_ratio': hyg_lqd_ratio,
                'change_30d': ratio_change,
                'risk_level': credit_risk,
                'credit_score': credit_score
            }
        
        # Reel Faiz Proxy: TIP performansı
        if not tip_hist.empty:
            tip_last = float(tip_hist['Close'].iloc[-1])
            tip_prev = float(tip_hist['Close'].iloc[-30]) if len(tip_hist) >= 30 else tip_last
            tip_change = ((tip_last - tip_prev) / tip_prev) * 100
            
            # TIP yükseliyorsa reel faiz düşüyor = BTC/Altın lehine
            if tip_change > 3:
                real_yield_trend = "DÜŞÜYOR"
                real_yield_score = 15
            elif tip_change < -3:
                real_yield_trend = "YÜKSELIYOR"
                real_yield_score = -10
            else:
                real_yield_trend = "STABIL"
                real_yield_score = 0
            
            results['real_yield'] = {
                'tip_value': tip_last,
                'change_30d': tip_change,
                'trend': real_yield_trend,
                'score': real_yield_score
            }
        
        # Copper/Gold Ratio: Ekonomik sağlık göstergesi
        if not copper_hist.empty and not gold_hist.empty:
            copper_last = float(copper_hist['Close'].iloc[-1])
            gold_last = float(gold_hist['Close'].iloc[-1])
            cu_au_ratio = copper_last / gold_last * 1000  # Normalize
            
            copper_prev = float(copper_hist['Close'].iloc[-30]) if len(copper_hist) >= 30 else copper_last
            gold_prev = float(gold_hist['Close'].iloc[-30]) if len(gold_hist) >= 30 else gold_last
            cu_au_prev = copper_prev / gold_prev * 1000
            
            cu_au_change = ((cu_au_ratio - cu_au_prev) / cu_au_prev) * 100
            
            # Cu/Au yükseliyorsa = ekonomik iyimserlik
            if cu_au_change > 5:
                economic_outlook = "İYİMSER"
                econ_score = 10
            elif cu_au_change < -5:
                economic_outlook = "KÖTÜMSER"
                econ_score = -10
            else:
                economic_outlook = "NÖTR"
                econ_score = 0
            
            results['copper_gold'] = {
                'ratio': cu_au_ratio,
                'change_30d': cu_au_change,
                'outlook': economic_outlook,
                'score': econ_score
            }
        
        return results, None
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_rolling_correlations(window: int = 30):
    """
    BTC ile diğer varlıklar arasındaki hareketli korelasyon.
    BTC 'teknoloji hissesi' mi yoksa 'dijital altın' mı gibi davranıyor?
    """
    import yfinance as yf
    import numpy as np
    
    try:
        btc = yf.Ticker('BTC-USD')
        nasdaq = yf.Ticker('^IXIC')
        gold = yf.Ticker('GC=F')
        sp500 = yf.Ticker('^GSPC')
        
        period = '6mo'
        btc_hist = btc.history(period=period)
        nasdaq_hist = nasdaq.history(period=period)
        gold_hist = gold.history(period=period)
        sp500_hist = sp500.history(period=period)
        
        if btc_hist.empty:
            return None, "BTC verisi alınamadı"
        
        # Tarihleri normalize et
        btc_ret = btc_hist['Close'].pct_change().dropna()
        btc_ret.index = btc_ret.index.date
        
        nasdaq_ret = nasdaq_hist['Close'].pct_change().dropna() if not nasdaq_hist.empty else None
        if nasdaq_ret is not None:
            nasdaq_ret.index = nasdaq_ret.index.date
        
        gold_ret = gold_hist['Close'].pct_change().dropna() if not gold_hist.empty else None
        if gold_ret is not None:
            gold_ret.index = gold_ret.index.date
        
        sp500_ret = sp500_hist['Close'].pct_change().dropna() if not sp500_hist.empty else None
        if sp500_ret is not None:
            sp500_ret.index = sp500_ret.index.date
        
        # Rolling correlation hesapla
        correlations = {'dates': [], 'btc_nasdaq': [], 'btc_gold': [], 'btc_sp500': []}
        
        # Ortak tarihleri bul
        all_dates = sorted(set(btc_ret.index))
        
        for i in range(window, len(all_dates)):
            date_window = all_dates[i-window:i]
            current_date = all_dates[i-1]
            
            try:
                btc_window = btc_ret.loc[[d for d in date_window if d in btc_ret.index]]
                
                if len(btc_window) < window // 2:
                    continue
                
                correlations['dates'].append(current_date)
                
                # BTC-Nasdaq korelasyonu
                if nasdaq_ret is not None:
                    nasdaq_window = nasdaq_ret.loc[[d for d in date_window if d in nasdaq_ret.index]]
                    common = btc_window.index.intersection(nasdaq_window.index)
                    if len(common) >= 10:
                        corr = btc_window.loc[common].corr(nasdaq_window.loc[common])
                        correlations['btc_nasdaq'].append(float(corr) if not np.isnan(corr) else 0)
                    else:
                        correlations['btc_nasdaq'].append(0)
                
                # BTC-Gold korelasyonu
                if gold_ret is not None:
                    gold_window = gold_ret.loc[[d for d in date_window if d in gold_ret.index]]
                    common = btc_window.index.intersection(gold_window.index)
                    if len(common) >= 10:
                        corr = btc_window.loc[common].corr(gold_window.loc[common])
                        correlations['btc_gold'].append(float(corr) if not np.isnan(corr) else 0)
                    else:
                        correlations['btc_gold'].append(0)
                
                # BTC-S&P500 korelasyonu
                if sp500_ret is not None:
                    sp_window = sp500_ret.loc[[d for d in date_window if d in sp500_ret.index]]
                    common = btc_window.index.intersection(sp_window.index)
                    if len(common) >= 10:
                        corr = btc_window.loc[common].corr(sp_window.loc[common])
                        correlations['btc_sp500'].append(float(corr) if not np.isnan(corr) else 0)
                    else:
                        correlations['btc_sp500'].append(0)
            except:
                continue
        
        # Son korelasyonlar
        if correlations['btc_nasdaq']:
            last_nasdaq_corr = correlations['btc_nasdaq'][-1]
        else:
            last_nasdaq_corr = 0
        
        if correlations['btc_gold']:
            last_gold_corr = correlations['btc_gold'][-1]
        else:
            last_gold_corr = 0
        
        # BTC karakteri belirleme
        if last_nasdaq_corr > 0.5:
            btc_character = "📈 Teknoloji Hissesi"
            character_detail = "BTC şu an Nasdaq ile yüksek korelasyonda"
        elif last_gold_corr > 0.3:
            btc_character = "🥇 Dijital Altın"
            character_detail = "BTC şu an altın ile pozitif korelasyonda"
        elif last_nasdaq_corr < 0 and last_gold_corr > 0:
            btc_character = "⚡ Bağımsız Varlık"
            character_detail = "BTC kendi dinamiğinde hareket ediyor"
        else:
            btc_character = "🔄 Geçiş Dönemi"
            character_detail = "BTC karakteri belirsiz"
        
        return {
            'history': correlations,
            'last_nasdaq_corr': last_nasdaq_corr,
            'last_gold_corr': last_gold_corr,
            'btc_character': btc_character,
            'character_detail': character_detail
        }, None
    except Exception as e:
        return None, str(e)


def prepare_master_features(macro_data, liquidity_data, yield_data, credit_data, fng_data, correlation_data):
    """
    XGBoost modeli için master feature matrix hazırlar.
    Tüm makro ve sentiment verilerini birleştirir.
    NaN değerlerini forward-fill ile doldurur.
    """
    import pandas as pd
    import numpy as np
    
    features = {}
    
    # Makro veriler
    if macro_data:
        features['dxy'] = macro_data.get('DXY', {}).get('value')
        features['dxy_change_5d'] = macro_data.get('DXY', {}).get('change_5d')
        features['vix'] = macro_data.get('VIX', {}).get('value')
        features['vix_change_5d'] = macro_data.get('VIX', {}).get('change_5d')
        features['gold_change_30d'] = macro_data.get('Gold', {}).get('change_30d')
        features['oil_change_5d'] = macro_data.get('Oil', {}).get('change_5d')
        features['usdjpy'] = macro_data.get('USDJPY', {}).get('value')
    
    # Likidite
    if liquidity_data:
        features['liquidity_score'] = liquidity_data.get('liquidity_score')
        features['tlt_change_30d'] = liquidity_data.get('tlt_change_30d')
    
    # Getiri eğrisi
    if yield_data:
        features['yield_spread'] = yield_data.get('spread')
        features['yield_inverted'] = 1 if yield_data.get('inverted') else 0
    
    # Kredi
    if credit_data:
        features['credit_score'] = credit_data.get('credit', {}).get('credit_score')
        features['real_yield_score'] = credit_data.get('real_yield', {}).get('score')
        features['copper_gold_score'] = credit_data.get('copper_gold', {}).get('score')
    
    # Sentiment
    if fng_data:
        features['fear_greed'] = fng_data.get('value')
        features['fear_greed_avg_7d'] = fng_data.get('avg_7d')
    
    # Korelasyonlar
    if correlation_data:
        features['btc_nasdaq_corr'] = correlation_data.get('last_nasdaq_corr')
        features['btc_gold_corr'] = correlation_data.get('last_gold_corr')
    
    # NaN değerlerini temizle (0 ile doldur)
    for key in features:
        if features[key] is None or (isinstance(features[key], float) and np.isnan(features[key])):
            features[key] = 0.0
        else:
            features[key] = float(features[key])
    
    # Session state'e kaydet
    st.session_state['master_features'] = features
    
    return features


@st.cache_data(ttl=43200, show_spinner=False)  # 12 saat cache
def fetch_geopolitical_trade_data():
    """
    Jeopolitik ve Ticaret Verileri.
    
    FRED API olmadan proxy'ler:
    - GPR Proxy: VIX volatilite + Altın volatilite kombinasyonu
    - BDI Proxy: BDRY ETF (Breakwave Dry Bulk Shipping)
    - Bank Stress: KBE (Bank ETF) / TLT oranı
    """
    import yfinance as yf
    import numpy as np
    
    try:
        results = {}
        
        # ===== GPR (Jeopolitik Risk) Proxy =====
        # VIX yüksek + Altın yükseliyor = Jeopolitik stres
        vix = yf.Ticker('^VIX')
        gold = yf.Ticker('GC=F')
        
        vix_hist = vix.history(period='3mo')
        gold_hist = gold.history(period='3mo')
        
        if not vix_hist.empty and not gold_hist.empty:
            vix_vol = float(vix_hist['Close'].std())
            vix_current = float(vix_hist['Close'].iloc[-1])
            vix_avg = float(vix_hist['Close'].mean())
            
            gold_ret = gold_hist['Close'].pct_change().dropna()
            gold_vol = float(gold_ret.std() * 100)
            
            # GPR Skoru: VIX seviyesi + Altın volatilitesi
            gpr_score = (vix_current / 20) * 50 + gold_vol * 10  # 0-100 arası normalize
            gpr_score = min(100, max(0, gpr_score))
            
            if gpr_score > 70:
                gpr_level = "YÜKSEK"
                gpr_risk_score = -15
            elif gpr_score > 50:
                gpr_level = "ORTA"
                gpr_risk_score = -5
            else:
                gpr_level = "DÜŞÜK"
                gpr_risk_score = 5
            
            results['gpr'] = {
                'score': gpr_score,
                'level': gpr_level,
                'risk_score': gpr_risk_score,
                'vix_current': vix_current,
                'vix_avg': vix_avg
            }
        
        # ===== Baltic Dry Index Proxy =====
        # BDRY ETF veya alternatif olarak nakliye şirketleri
        try:
            bdry = yf.Ticker('BDRY')  # Baltic Dry ETF
            bdry_hist = bdry.history(period='6mo')
            
            if not bdry_hist.empty:
                bdry_last = float(bdry_hist['Close'].iloc[-1])
                bdry_prev = float(bdry_hist['Close'].iloc[-30]) if len(bdry_hist) >= 30 else bdry_last
                bdry_change = ((bdry_last - bdry_prev) / bdry_prev) * 100
                
                if bdry_change > 10:
                    trade_outlook = "CANLI"
                    trade_score = 10
                elif bdry_change < -10:
                    trade_outlook = "DURGUN"
                    trade_score = -10
                else:
                    trade_outlook = "NORMAL"
                    trade_score = 0
                
                results['trade'] = {
                    'bdi_value': bdry_last,
                    'change_30d': bdry_change,
                    'outlook': trade_outlook,
                    'score': trade_score
                }
        except:
            pass
        
        # ===== Bank Stress Proxy =====
        # KBE (Bank ETF) / TLT (Treasury ETF) oranı
        try:
            kbe = yf.Ticker('KBE')  # SPDR S&P Bank ETF
            tlt = yf.Ticker('TLT')  # Long Treasury ETF
            
            kbe_hist = kbe.history(period='3mo')
            tlt_hist = tlt.history(period='3mo')
            
            if not kbe_hist.empty and not tlt_hist.empty:
                kbe_last = float(kbe_hist['Close'].iloc[-1])
                tlt_last = float(tlt_hist['Close'].iloc[-1])
                bank_ratio = kbe_last / tlt_last
                
                kbe_prev = float(kbe_hist['Close'].iloc[-30]) if len(kbe_hist) >= 30 else kbe_last
                tlt_prev = float(tlt_hist['Close'].iloc[-30]) if len(tlt_hist) >= 30 else tlt_last
                prev_ratio = kbe_prev / tlt_prev
                
                ratio_change = ((bank_ratio - prev_ratio) / prev_ratio) * 100
                
                # Oran düşüyorsa = bankalar tahvillere göre zayıflıyor = stres
                if ratio_change < -5:
                    bank_stress = "YÜKSEK"
                    bank_score = -20
                elif ratio_change > 5:
                    bank_stress = "DÜŞÜK"
                    bank_score = 10
                else:
                    bank_stress = "NORMAL"
                    bank_score = 0
                
                results['bank'] = {
                    'kbe_tlt_ratio': bank_ratio,
                    'change_30d': ratio_change,
                    'stress_level': bank_stress,
                    'score': bank_score
                }
        except:
            pass
        
        # ===== Varlık Rotasyonu Rasyoları =====
        try:
            nasdaq = yf.Ticker('^IXIC')
            btc = yf.Ticker('BTC-USD')
            dxy = yf.Ticker('DX-Y.NYB')
            
            nasdaq_hist = nasdaq.history(period='3mo')
            btc_hist = btc.history(period='3mo')
            dxy_hist = dxy.history(period='3mo')
            
            ratios = {}
            
            # Nasdaq/Gold Oranı
            if not nasdaq_hist.empty and not gold_hist.empty:
                nasdaq_last = float(nasdaq_hist['Close'].iloc[-1])
                gold_last = float(gold_hist['Close'].iloc[-1])
                nq_gold = nasdaq_last / gold_last
                
                nasdaq_prev = float(nasdaq_hist['Close'].iloc[-30]) if len(nasdaq_hist) >= 30 else nasdaq_last
                gold_prev = float(gold_hist['Close'].iloc[-30]) if len(gold_hist) >= 30 else gold_last
                nq_gold_prev = nasdaq_prev / gold_prev
                
                nq_gold_change = ((nq_gold - nq_gold_prev) / nq_gold_prev) * 100
                
                if nq_gold_change > 5:
                    rotation = "RISK-ON"
                elif nq_gold_change < -5:
                    rotation = "RISK-OFF"
                else:
                    rotation = "NÖTR"
                
                ratios['nasdaq_gold'] = {
                    'ratio': nq_gold,
                    'change_30d': nq_gold_change,
                    'rotation': rotation
                }
            
            # BTC/DXY Oranı
            if not btc_hist.empty and not dxy_hist.empty:
                btc_last = float(btc_hist['Close'].iloc[-1])
                dxy_last = float(dxy_hist['Close'].iloc[-1])
                btc_dxy = btc_last / dxy_last
                
                btc_prev = float(btc_hist['Close'].iloc[-30]) if len(btc_hist) >= 30 else btc_last
                dxy_prev = float(dxy_hist['Close'].iloc[-30]) if len(dxy_hist) >= 30 else dxy_last
                btc_dxy_prev = btc_prev / dxy_prev
                
                btc_dxy_change = ((btc_dxy - btc_dxy_prev) / btc_dxy_prev) * 100
                
                ratios['btc_dxy'] = {
                    'ratio': btc_dxy,
                    'change_30d': btc_dxy_change
                }
            
            results['ratios'] = ratios
        except:
            pass
        
        return results, None
    except Exception as e:
        return None, str(e)


def prepare_master_features_final(base_features: dict, geo_data: dict = None) -> dict:
    """
    XGBoost için final feature matrix.
    Tüm verileri birleştirir ve NaN temizliği yapar.
    """
    import numpy as np
    
    features = base_features.copy() if base_features else {}
    
    # Jeopolitik ve ticaret verileri
    if geo_data:
        if geo_data.get('gpr'):
            features['gpr_score'] = geo_data['gpr']['score']
            features['gpr_risk_score'] = geo_data['gpr']['risk_score']
        
        if geo_data.get('trade'):
            features['bdi_change'] = geo_data['trade']['change_30d']
            features['trade_score'] = geo_data['trade']['score']
        
        if geo_data.get('bank'):
            features['bank_stress_score'] = geo_data['bank']['score']
            features['kbe_tlt_change'] = geo_data['bank']['change_30d']
        
        if geo_data.get('ratios'):
            if geo_data['ratios'].get('nasdaq_gold'):
                features['nasdaq_gold_change'] = geo_data['ratios']['nasdaq_gold']['change_30d']
            if geo_data['ratios'].get('btc_dxy'):
                features['btc_dxy_change'] = geo_data['ratios']['btc_dxy']['change_30d']
    
    # NaN temizliği ve tip dönüşümü
    cleaned = {}
    for key, value in features.items():
        if value is None:
            cleaned[key] = 0.0
        elif isinstance(value, (int, float)):
            if np.isnan(value) or np.isinf(value):
                cleaned[key] = 0.0
            else:
                cleaned[key] = float(value)
        else:
            try:
                cleaned[key] = float(value)
            except:
                cleaned[key] = 0.0
    
    # Session state'e kaydet
    st.session_state['master_features_final'] = cleaned
    
    return cleaned


@st.cache_data(ttl=21600, show_spinner=False)  # 6 saat cache
def fetch_fear_greed_index():
    """
    Crypto Fear & Greed Index (Alternative.me API).
    0-24: Extreme Fear
    25-49: Fear
    50-74: Greed
    75-100: Extreme Greed
    """
    try:
        url = "https://api.alternative.me/fng/?limit=30"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            fng_data = data.get('data', [])
            
            if fng_data:
                current = fng_data[0]
                value = int(current.get('value', 50))
                classification = current.get('value_classification', 'Neutral')
                
                # 7 günlük ortalama
                if len(fng_data) >= 7:
                    avg_7d = sum(int(d['value']) for d in fng_data[:7]) / 7
                else:
                    avg_7d = value
                
                return {
                    'value': value,
                    'classification': classification,
                    'avg_7d': avg_7d,
                    'history': [{'date': d['timestamp'], 'value': int(d['value'])} for d in fng_data]
                }, None
        return None, "API yanıt vermedi"
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_market_sentiment():
    """
    Piyasa duyarlılık göstergeleri.
    VIX ve SKEW kullanarak piyasa stresini ölçer.
    """
    import yfinance as yf
    
    try:
        vix = yf.Ticker('^VIX')
        # SKEW: Tail risk göstergesi
        
        vix_hist = vix.history(period='30d')
        
        if vix_hist.empty:
            return None, "VIX verisi alınamadı"
        
        vix_current = float(vix_hist['Close'].iloc[-1])
        vix_avg = float(vix_hist['Close'].mean())
        vix_high = float(vix_hist['Close'].max())
        
        # Sentiment skoru (0-100, yüksek = olumlu)
        if vix_current < 15:
            sentiment_score = 85
            sentiment_label = "Aşırı İyimser"
        elif vix_current < 20:
            sentiment_score = 70
            sentiment_label = "İyimser"
        elif vix_current < 25:
            sentiment_score = 50
            sentiment_label = "Nötr"
        elif vix_current < 30:
            sentiment_score = 30
            sentiment_label = "Endişeli"
        else:
            sentiment_score = 15
            sentiment_label = "Panik"
        
        return {
            'vix_current': vix_current,
            'vix_avg_30d': vix_avg,
            'vix_high_30d': vix_high,
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label
        }, None
    except Exception as e:
        return None, str(e)


def analyze_market_regime(macro_data: dict, liquidity_data: dict = None, 
                          yield_data: dict = None, sentiment_data: dict = None,
                          fng_data: dict = None) -> dict:
    """
    Piyasa Rejimi Analizi - 4 Rejim Modeli.
    
    Rejim 1: Enflasyonist Büyüme (Kripto/Hisse Dostu)
        - Likidite artıyor, VIX düşük, DXY zayıf
    
    Rejim 2: Stagflasyon (Altın Dostu, Kripto Riskli)
        - Yüksek enflasyon + düşük büyüme
        
    Rejim 3: Deflasyonist Resesyon (Nakit/Tahvil Dostu)
        - Getiri eğrisi ters, VIX yüksek
        
    Rejim 4: Goldilocks (Her Şey İyi)
        - Düşük enflasyon, normal büyüme, likidite bol
    
    Returns:
        dict: regime, description, best_asset, confidence
    """
    scores = {
        'growth': 0,      # Büyüme skoru
        'inflation': 0,   # Enflasyon baskısı
        'liquidity': 0,   # Likidite durumu
        'risk': 0         # Risk iştahı
    }
    
    # Likidite analizi
    if liquidity_data:
        liq_trend = liquidity_data.get('liquidity_trend', 'STABIL')
        if liq_trend == "ARTIYOR":
            scores['liquidity'] += 30
            scores['growth'] += 20
        elif liq_trend == "AZALIYOR":
            scores['liquidity'] -= 30
            scores['growth'] -= 10
    
    # Getiri eğrisi analizi
    if yield_data:
        if yield_data.get('inverted', False):
            scores['growth'] -= 40  # Resesyon sinyali
            scores['risk'] -= 30
        elif yield_data.get('spread', 1) < 0.5:
            scores['growth'] -= 20
    
    # DXY analizi
    if macro_data.get('DXY'):
        dxy_val = macro_data['DXY']['value']
        if dxy_val > 105:
            scores['liquidity'] -= 20
            scores['inflation'] += 10
        elif dxy_val < 100:
            scores['liquidity'] += 20
            scores['risk'] += 15
    
    # VIX analizi
    if macro_data.get('VIX'):
        vix_val = macro_data['VIX']['value']
        if vix_val > 30:
            scores['risk'] -= 40
        elif vix_val < 20:
            scores['risk'] += 30
    
    # Altın analizi (enflasyon proxy)
    if macro_data.get('Gold'):
        gold_change = macro_data['Gold'].get('change_30d', 0)
        if gold_change > 5:
            scores['inflation'] += 25
        elif gold_change < -5:
            scores['inflation'] -= 15
    
    # Fear & Greed
    if fng_data:
        fng_val = fng_data.get('value', 50)
        if fng_val > 70:
            scores['risk'] += 20
        elif fng_val < 30:
            scores['risk'] -= 20
    
    # Rejim belirleme
    total_growth = scores['growth'] + scores['liquidity']
    total_risk = scores['risk']
    inflation_pressure = scores['inflation']
    
    if total_growth > 30 and total_risk > 20 and inflation_pressure < 20:
        regime = "GOLDILOCKS"
        description = "Goldilocks: Düşük enflasyon, sağlıklı büyüme, bol likidite"
        best_asset = "🪙 Kripto & 📈 Hisse"
        color = "#00C853"
        confidence = min(90, 50 + total_growth // 2)
    elif total_growth > 20 and inflation_pressure > 15:
        regime = "ENFLASYONIST BÜYÜME"
        description = "Enflasyonist Büyüme: Likidite bol ama enflasyon baskısı var"
        best_asset = "🪙 Kripto & 🥇 Altın"
        color = "#FF9800"
        confidence = min(85, 50 + total_growth // 3)
    elif inflation_pressure > 25 and total_growth < 0:
        regime = "STAGFLASYON"
        description = "Stagflasyon: Yüksek enflasyon + düşük büyüme - en kötü senaryo"
        best_asset = "🥇 Altın & 💵 Nakit"
        color = "#FF5722"
        confidence = min(80, 40 + inflation_pressure)
    elif total_growth < -20 or (yield_data and yield_data.get('inverted')):
        regime = "RESESYON RİSKİ"
        description = "Deflasyonist Resesyon: Getiri eğrisi ters, büyüme yavaşlıyor"
        best_asset = "📜 Tahvil & 💵 Nakit"
        color = "#FF1744"
        confidence = min(85, 60 - total_growth // 2)
    else:
        regime = "KARIŞIK SİNYALLER"
        description = "Geçiş Dönemi: Piyasa yön arıyor, dikkatli olun"
        best_asset = "⚖️ Dengeli Portföy"
        color = "#9E9E9E"
        confidence = 50
    
    # Session state'e kaydet
    st.session_state['market_regime'] = regime
    st.session_state['feature_matrix'] = {
        'scores': scores,
        'regime': regime,
        'dxy': macro_data.get('DXY', {}).get('value'),
        'vix': macro_data.get('VIX', {}).get('value'),
        'gold_change': macro_data.get('Gold', {}).get('change_30d'),
        'liquidity_trend': liquidity_data.get('liquidity_trend') if liquidity_data else None,
        'yield_spread': yield_data.get('spread') if yield_data else None,
        'fng': fng_data.get('value') if fng_data else None
    }
    
    return {
        'regime': regime,
        'description': description,
        'best_asset': best_asset,
        'color': color,
        'confidence': confidence,
        'scores': scores
    }


def calculate_risk_score(macro_data: dict, liquidity_data: dict = None, yield_data: dict = None) -> tuple:
    """
    Gelişmiş Risk İştahı Skoru (0-100) hesaplar.
    
    RISK-ON faktörler (skoru artırır):
    - DXY düşük (<100) → Zayıf dolar, likidite bol
    - VIX düşük (<20) → Piyasa sakin
    - Net Likidite artıyor → Fed gevşiyor
    - Petrol yükseliyor → Ekonomik aktivite güçlü
    
    RISK-OFF faktörler (skoru düşürür):
    - VIX yüksek (>30) → Korku yüksek
    - JPY güçleniyor → Carry trade çözülüyor
    - Getiri eğrisi tersine dönmüş → Resesyon riski
    - Altın yükseliyor → Güvenli liman talebi
    
    Returns:
        (score, factors, alerts): Skor, faktör listesi ve kritik uyarılar
    """
    score = 50  # Nötr başla
    factors = []
    alerts = []  # Kritik uyarılar
    
    # ==================== LİKİDİTE ANALİZİ (+/-20) ====================
    if liquidity_data:
        liq_score = liquidity_data.get('liquidity_score', 0)
        liq_trend = liquidity_data.get('liquidity_trend', 'STABIL')
        tlt_change = liquidity_data.get('tlt_change_30d', 0)
        
        score += liq_score
        
        if liq_trend == "ARTIYOR":
            factors.append(("🟢 Likidite Artıyor", f"TLT: +{tlt_change:.1f}% (Fed gevşiyor)"))
        elif liq_trend == "AZALIYOR":
            factors.append(("🔴 Likidite Azalıyor", f"TLT: {tlt_change:.1f}% (Fed sıkılaştırıyor)"))
            alerts.append("⚠️ Likidite daralıyor - riskli varlıklar baskı altında")
        else:
            factors.append(("🟡 Likidite Stabil", f"TLT: {tlt_change:+.1f}%"))
    
    # ==================== GETİRİ EĞRİSİ ANALİZİ (+/-15) ====================
    if yield_data:
        spread = yield_data.get('spread', 0)
        inverted = yield_data.get('inverted', False)
        
        if inverted:
            score -= 15
            factors.append(("🔴 Getiri Eğrisi Ters", f"Spread: {spread:.2f}% (10Y < 2Y)"))
            alerts.append("🚨 RESESYON ALARMI: Getiri eğrisi tersine döndü!")
        elif spread < 0.5:
            score -= 5
            factors.append(("🟡 Düzleşen Eğri", f"Spread: {spread:.2f}% (Dikkat)"))
        else:
            score += 10
            factors.append(("🟢 Normal Eğri", f"Spread: {spread:.2f}%"))
    
    # ==================== DXY ANALİZİ (+/-15) ====================
    dxy = macro_data.get('DXY')
    if dxy:
        dxy_val = dxy['value']
        if dxy_val < 100:
            score += 15
            factors.append(("🟢 Zayıf Dolar", f"DXY: {dxy_val:.1f} < 100"))
        elif dxy_val > 105:
            score -= 15
            factors.append(("🔴 Güçlü Dolar", f"DXY: {dxy_val:.1f} > 105"))
        else:
            factors.append(("🟡 Nötr Dolar", f"DXY: {dxy_val:.1f}"))
    
    # ==================== VIX ANALİZİ (+/-20) ====================
    vix = macro_data.get('VIX')
    if vix:
        vix_val = vix['value']
        if vix_val < 15:
            score += 20
            factors.append(("🟢 Düşük Korku", f"VIX: {vix_val:.1f} < 15"))
        elif vix_val < 20:
            score += 10
            factors.append(("🟢 Normal Korku", f"VIX: {vix_val:.1f}"))
        elif vix_val > 30:
            score -= 20
            factors.append(("🔴 Yüksek Korku", f"VIX: {vix_val:.1f} > 30"))
            alerts.append("⚠️ VIX 30 üzerinde - volatilite yüksek")
        elif vix_val > 25:
            score -= 10
            factors.append(("🟡 Artan Korku", f"VIX: {vix_val:.1f}"))
        else:
            factors.append(("🟡 Orta Korku", f"VIX: {vix_val:.1f}"))
    
    # ==================== CARRY TRADE / YEN ANALİZİ (+/-10) ====================
    usdjpy = macro_data.get('USDJPY')
    if usdjpy:
        jpy_val = usdjpy['value']
        jpy_change = usdjpy.get('change_5d', 0)
        
        if jpy_val > 155:
            score += 10
            factors.append(("🟢 Zayıf Yen", f"USD/JPY: {jpy_val:.1f} (Carry Trade aktif)"))
        elif jpy_val < 145:
            score -= 10
            factors.append(("🔴 Güçlü Yen", f"USD/JPY: {jpy_val:.1f} (Carry Trade çözülüyor)"))
            if jpy_change < -2:
                alerts.append("⚠️ Yen hızla güçleniyor - carry trade riski")
        else:
            factors.append(("🟡 Stabil Yen", f"USD/JPY: {jpy_val:.1f}"))
    
    # ==================== EMTİA ANALİZİ (+/-5) ====================
    oil = macro_data.get('Oil')
    if oil:
        oil_change = oil.get('change_5d', 0)
        if oil_change > 5:
            score += 5
            factors.append(("🟢 Petrol Yükseliyor", f"+{oil_change:.1f}% (Ekonomik aktivite)"))
        elif oil_change < -5:
            score -= 5
            factors.append(("🔴 Petrol Düşüyor", f"{oil_change:.1f}% (Talep endişesi)"))
    
    gold = macro_data.get('Gold')
    if gold:
        gold_change = gold.get('change_5d', 0)
        if gold_change > 3:
            score -= 5
            factors.append(("🔴 Altın Yükseliyor", f"+{gold_change:.1f}% (Risk-off sinyali)"))
        elif gold_change < -3:
            score += 5
            factors.append(("🟢 Altın Düşüyor", f"{gold_change:.1f}% (Risk-on sinyali)"))
    
    # Session state'e kaydet
    st.session_state['risk_score'] = max(0, min(100, score))
    st.session_state['risk_alerts'] = alerts
    
    return max(0, min(100, score)), factors, alerts


@st.cache_data(ttl=600, show_spinner=False)
def fetch_correlation_heatmap_data(days: int = 30):
    """Varlıklar arası korelasyon matrisi için veri çeker."""
    import yfinance as yf
    import numpy as np
    
    assets = {
        'BTC': 'BTC-USD',
        'ETH': 'ETH-USD',
        'DXY': 'DX-Y.NYB',
        'VIX': '^VIX',
        'Gold': 'GC=F',
        'Oil': 'CL=F',
        'JPY': 'JPY=X',
        'SP500': '^GSPC'
    }
    
    try:
        returns_data = {}
        
        for name, symbol in assets.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=f'{days}d')
                
                if not hist.empty and len(hist) > 5:
                    # Günlük getiri
                    returns = hist['Close'].pct_change().dropna()
                    returns.index = returns.index.date
                    returns_data[name] = returns
            except:
                continue
        
        if len(returns_data) < 3:
            return None, "Yeterli veri yok"
        
        # DataFrame oluştur
        import pandas as pd
        df = pd.DataFrame(returns_data)
        
        # Korelasyon matrisi
        corr_matrix = df.corr()
        
        return corr_matrix, None
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=600, show_spinner=False)
def fetch_correlation_data(crypto_symbol: str = "BTC-USD", days: int = 90):
    """DXY ve Kripto arasındaki korelasyonu hesaplar."""
    import yfinance as yf
    import numpy as np
    
    try:
        dxy = yf.Ticker('DX-Y.NYB')
        crypto = yf.Ticker(crypto_symbol)
        
        dxy_hist = dxy.history(period=f'{days}d')
        crypto_hist = crypto.history(period=f'{days}d')
        
        if dxy_hist.empty or crypto_hist.empty:
            return None, "Veri yetersiz"
        
        # DataFrame'leri hazırla - sadece Close kolonunu al
        dxy_df = dxy_hist[['Close']].copy()
        dxy_df.columns = ['DXY']
        dxy_df.index = dxy_df.index.date  # Sadece tarih, saat yok
        
        crypto_df = crypto_hist[['Close']].copy()
        crypto_df.columns = ['Crypto']
        crypto_df.index = crypto_df.index.date
        
        # İç birleştirme - ortak tarihleri bul
        merged = dxy_df.join(crypto_df, how='inner')
        
        if len(merged) < 10:
            return None, f"Yeterli ortak gün yok ({len(merged)} gün)"
        
        # Getiri hesapla
        merged['DXY_ret'] = merged['DXY'].pct_change()
        merged['Crypto_ret'] = merged['Crypto'].pct_change()
        merged = merged.dropna()
        
        if len(merged) < 5:
            return None, "Yeterli getiri verisi yok"
        
        # Korelasyon hesapla
        correlation = merged['DXY_ret'].corr(merged['Crypto_ret'])
        
        return {
            'correlation': correlation,
            'dxy_data': dxy_hist,
            'crypto_data': crypto_hist,
            'days': f"{len(merged)} gün"
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


# ==================== ML-READY FEATURE ENGINEERING ====================

def calculate_rsi(prices, period=14):
    """RSI (Relative Strength Index) hesaplar."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def prepare_ml_features(price_df, macro_df=None):
    """
    XGBoost/ML modeli için feature hazırlığı.
    
    Bu fonksiyon gelecekteki ML entegrasyonu için temel oluşturur.
    
    Args:
        price_df: OHLCV verisi (timestamp, open, high, low, close, volume)
        macro_df: Makro veriler (opsiyonel - DXY, bonds vb.)
    
    Returns:
        DataFrame: ML modeli için hazır feature seti
    
    Features:
    - Price: close, returns, log_returns
    - Technical: RSI_14, EMA_20, EMA_50, EMA_200, volatility_20
    - Macro: DXY, DXY_change (eğer sağlanırsa)
    """
    if price_df is None or price_df.empty:
        return None
    
    features = price_df.copy()
    
    # Fiyat bazlı özellikler
    features['returns'] = features['close'].pct_change()
    features['log_returns'] = features['close'].apply(lambda x: x if x <= 0 else x).transform(lambda x: x.pct_change())
    
    # Volatilite (20 günlük)
    features['volatility_20'] = features['returns'].rolling(window=20).std()
    
    # Teknik indikatörler
    features['RSI_14'] = calculate_rsi(features['close'], 14)
    features['EMA_20'] = features['close'].ewm(span=20, adjust=False).mean()
    features['EMA_50'] = features['close'].ewm(span=50, adjust=False).mean()
    features['EMA_200'] = features['close'].ewm(span=200, adjust=False).mean()
    
    # EMA sinyalleri (binary)
    features['above_EMA_20'] = (features['close'] > features['EMA_20']).astype(int)
    features['above_EMA_50'] = (features['close'] > features['EMA_50']).astype(int)
    features['above_EMA_200'] = (features['close'] > features['EMA_200']).astype(int)
    
    # Makro veriler (opsiyonel)
    if macro_df is not None and not macro_df.empty:
        # Tarihleri normalize et
        features['date'] = features['timestamp'].dt.date if 'timestamp' in features.columns else features.index.date
        macro_df['date'] = macro_df.index.date if hasattr(macro_df.index, 'date') else macro_df.index
        
        # Merge
        features = features.merge(macro_df[['date', 'DXY']], on='date', how='left')
        features['DXY_change'] = features['DXY'].pct_change()
    
    return features.dropna()


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
    if treasury_data and isinstance(treasury_data, dict):
        total_treasury = 0.0
        
        # Farklı treasury formatlarını dene
        raw_tvl = treasury_data.get('tvl', 0)
        
        if isinstance(raw_tvl, (int, float)) and raw_tvl > 0:
            total_treasury = float(raw_tvl)
        else:
            # tokenBreakdowns veya ownTokens içinden topla
            token_breakdowns = treasury_data.get('tokenBreakdowns', {})
            if token_breakdowns and isinstance(token_breakdowns, dict):
                for chain_data in token_breakdowns.values():
                    if isinstance(chain_data, dict):
                        for token_data in chain_data.values():
                            if isinstance(token_data, dict):
                                total_treasury += float(token_data.get('usdValue', 0) or 0)
                            elif isinstance(token_data, (int, float)):
                                total_treasury += float(token_data)
            
            # ownTokens kontrolü
            own_tokens = treasury_data.get('ownTokens', 0)
            if isinstance(own_tokens, (int, float)):
                total_treasury += float(own_tokens)
        
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
            details.append("⚪ Hazine verisi mevcut değil")
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
    
    st.divider()
    
    # Piyasa Riski (DXY bazlı)
    st.subheader("🌡️ Piyasa Riski (Buffett Pusulası)")
    
    macro_data = fetch_macro_data()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if macro_data.get('DXY'):
            dxy_val = macro_data['DXY']['value']
            dxy_change = macro_data['DXY']['change']
            
            if dxy_val > 105:
                risk_level = "🔴 Yüksek Risk"
                risk_color = "#FF1744"
            elif dxy_val > 100:
                risk_level = "🟡 Orta Risk"
                risk_color = "#FF9800"
            else:
                risk_level = "🟢 Düşük Risk"
                risk_color = "#00C853"
            
            st.metric(f"💵 DXY ({risk_level})", f"{dxy_val:.2f}", f"{dxy_change:+.2f}%")
        else:
            st.metric("💵 DXY", "—")
    
    with col2:
        if macro_data.get('VIX'):
            vix_val = macro_data['VIX']['value']
            vix_change = macro_data['VIX']['change']
            
            vix_status = "🟢" if vix_val < 20 else "🟡" if vix_val < 30 else "🔴"
            st.metric(f"😱 VIX {vix_status}", f"{vix_val:.1f}", f"{vix_change:+.2f}%")
        else:
            st.metric("😱 VIX", "—")
    
    with col3:
        if macro_data.get('US10Y'):
            bond_val = macro_data['US10Y']['value']
            bond_change = macro_data['US10Y']['change']
            st.metric("📜 ABD 10Y", f"%{bond_val:.2f}", f"{bond_change:+.2f}%")
        else:
            st.metric("📜 ABD 10Y", "—")


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
        # TVL birden fazla formatta gelebilir
        raw_tvl = protocol_data.get('tvl', 0)
        
        if isinstance(raw_tvl, list) and len(raw_tvl) > 0:
            # Liste formatı - son değeri al
            last_item = raw_tvl[-1]
            if isinstance(last_item, dict):
                tvl = float(last_item.get('totalLiquidityUSD', 0) or last_item.get('tvl', 0) or 0)
            else:
                tvl = float(last_item) if last_item else 0.0
        elif isinstance(raw_tvl, (int, float)):
            tvl = float(raw_tvl)
        else:
            # currentChainTvls'den topla
            current_tvls = protocol_data.get('currentChainTvls', {})
            if current_tvls and isinstance(current_tvls, dict):
                tvl = sum(float(v) for v in current_tvls.values() if isinstance(v, (int, float)))
            else:
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
        
        # TVL Trendi - Basitleştirilmiş yaklaşım
        st.subheader("📈 TVL Geçmişi")
        
        try:
            # Doğrudan tvl dizisini kullan (chainTvls yerine)
            tvl_history = protocol_data.get('tvl', [])
            
            # Eğer tvl bir liste değilse, farklı formatlara bak
            if not isinstance(tvl_history, list):
                # Belki bir sayı olarak gelmiştir - geçmişi çekilemez
                tvl_history = []
            
            if tvl_history and len(tvl_history) > 5:
                # TVL history formatı: [{"date": timestamp, "totalLiquidityUSD": value}, ...]
                df_tvl = pd.DataFrame(tvl_history)
                
                # Farklı format kontrolleri
                if 'date' in df_tvl.columns:
                    df_tvl['date'] = pd.to_datetime(df_tvl['date'], unit='s')
                    
                    # Değer kolonunu bul
                    value_col = None
                    for col in ['totalLiquidityUSD', 'tvl', 'value']:
                        if col in df_tvl.columns:
                            value_col = col
                            break
                    
                    if value_col:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df_tvl['date'],
                            y=df_tvl[value_col],
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
                        st.info("TVL değer kolonu bulunamadı.")
                else:
                    st.info("TVL geçmiş formatı desteklenmiyor.")
            else:
                st.info("TVL geçmiş verisi bulunamadı veya yetersiz.")
        except Exception as e:
            st.info(f"TVL geçmişi yüklenemedi.")
        
        # Treasury Bilgisi
        if treasury_data:
            st.divider()
            st.subheader("💰 Hazine (Treasury) Durumu")
            
            raw_treasury = treasury_data.get('tvl', 0)
            if isinstance(raw_treasury, (int, float)):
                treasury_tvl = float(raw_treasury)
            elif isinstance(raw_treasury, dict):
                treasury_tvl = sum(float(v) for v in raw_treasury.values() if isinstance(v, (int, float)))
            else:
                treasury_tvl = 0
            
            if treasury_tvl > 0:
                st.metric("Toplam Hazine", f"${treasury_tvl/1e6:.1f}M")
            else:
                st.metric("Toplam Hazine", "Veri yok")
    else:
        st.error(f"Protokol verisi alınamadı: {proto_err}")
        st.info("💡 DeFiLlama API'sine bağlanırken sorun oluştu. Lütfen tekrar deneyin.")


def render_macro_page():
    """Makro Ekonomi Sayfası - Piyasa Pusulası v3"""
    st.title("📊 Makro Ekonomi - Piyasa Pusulası v3")
    st.caption("Likidite takibi, piyasa rejimi analizi ve yatırım karar desteği")
    st.divider()
    
    # Makro verileri çek (Lazy Loading)
    with st.spinner("Makro veriler yükleniyor..."):
        macro_data = fetch_macro_data()
        liquidity_data, liq_err = fetch_liquidity_proxy()
        yield_data, yield_err = fetch_yield_curve_data()
        fng_data, fng_err = fetch_fear_greed_index()
        sentiment_data, sent_err = fetch_market_sentiment()
        credit_data, credit_err = fetch_credit_and_liquidity_data()
        correlation_data, corr_err = fetch_rolling_correlations(30)
        geo_data, geo_err = fetch_geopolitical_trade_data()
    
    # Master features hazırla (XGBoost için)
    base_features = prepare_master_features(macro_data, liquidity_data, yield_data, credit_data, fng_data, correlation_data)
    master_features = prepare_master_features_final(base_features, geo_data)
    
    # ==================== PİYASA REJİMİ ====================
    st.subheader("🎯 Piyasa Rejimi Analizi")
    
    regime_analysis = analyze_market_regime(macro_data, liquidity_data, yield_data, sentiment_data, fng_data)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        regime_color = regime_analysis['color']
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, {regime_color}22, {regime_color}44); border-radius: 15px; border: 3px solid {regime_color};">
            <h2 style="color: {regime_color}; margin: 0; font-size: 1.3rem;">{regime_analysis['regime']}</h2>
            <p style="color: #888; margin: 10px 0; font-size: 0.9rem;">Güven: %{regime_analysis['confidence']}</p>
            <h3 style="color: {regime_color}; margin: 0;">En İyi Varlık:</h3>
            <h2 style="color: {regime_color}; margin: 5px 0;">{regime_analysis['best_asset']}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.info(regime_analysis['description'])
        
        # Skor detayları
        scores = regime_analysis['scores']
        with st.expander("📊 Rejim Skorları"):
            score_cols = st.columns(4)
            with score_cols[0]:
                st.metric("📈 Büyüme", f"{scores['growth']:+d}")
            with score_cols[1]:
                st.metric("💰 Likidite", f"{scores['liquidity']:+d}")
            with score_cols[2]:
                st.metric("🔥 Enflasyon", f"{scores['inflation']:+d}")
            with score_cols[3]:
                st.metric("⚡ Risk", f"{scores['risk']:+d}")
    
    st.divider()
    
    # ==================== FEAR & GREED ====================
    st.subheader("😱 Kripto Fear & Greed Index")
    
    if fng_data:
        fng_cols = st.columns([1, 2, 1])
        
        with fng_cols[0]:
            fng_val = fng_data['value']
            if fng_val < 25:
                fng_color = "#FF1744"
                fng_label = "Extreme Fear"
            elif fng_val < 45:
                fng_color = "#FF5722"
                fng_label = "Fear"
            elif fng_val < 55:
                fng_color = "#FF9800"
                fng_label = "Neutral"
            elif fng_val < 75:
                fng_color = "#8BC34A"
                fng_label = "Greed"
            else:
                fng_color = "#00C853"
                fng_label = "Extreme Greed"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: {fng_color}22; border-radius: 15px; border: 3px solid {fng_color};">
                <h1 style="color: {fng_color}; margin: 0; font-size: 3rem;">{fng_val}</h1>
                <p style="color: {fng_color}; margin: 0;">{fng_label}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with fng_cols[1]:
            # Fear & Greed grafiği
            if fng_data.get('history'):
                import pandas as pd
                fng_df = pd.DataFrame(fng_data['history'])
                fng_df['date'] = pd.to_datetime(fng_df['date'].astype(int), unit='s')
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=fng_df['date'],
                    y=fng_df['value'],
                    mode='lines+markers',
                    fill='tozeroy',
                    line=dict(color='#FF9800', width=2),
                    name='F&G Index'
                ))
                
                # Referans çizgileri
                fig.add_hline(y=25, line_dash="dash", line_color="red", annotation_text="Korku")
                fig.add_hline(y=75, line_dash="dash", line_color="green", annotation_text="Açgözlülük")
                
                fig.update_layout(
                    template="plotly_dark",
                    height=200,
                    margin=dict(l=0, r=0, t=10, b=0),
                    yaxis=dict(range=[0, 100])
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with fng_cols[2]:
            st.metric("7 Gün Ort.", f"{fng_data['avg_7d']:.0f}")
            if fng_val < 30:
                st.success("💡 Aşırı korku = Alım fırsatı olabilir")
            elif fng_val > 70:
                st.warning("💡 Aşırı açgözlülük = Dikkatli ol")
    else:
        st.warning(f"Fear & Greed verisi alınamadı: {fng_err}")
    
    st.divider()
    
    # ==================== RİSK PUSULASI ====================
    st.subheader("🧭 Risk Pusulası v2.0")
    
    risk_score, risk_factors, risk_alerts = calculate_risk_score(macro_data, liquidity_data, yield_data)
    
    # Kritik uyarılar varsa göster
    if risk_alerts:
        for alert in risk_alerts:
            st.error(alert)
    # Risk durumu kartı
    if risk_score > 70:
        risk_mode = "RISK-ON"
        risk_color = "#00C853"
        risk_message = "Piyasa RISK-ON modunda. Likidite artıyor, riskli varlıklar (Kripto/Hisse) için uygun ortam."
        risk_emoji = "🟢"
    elif risk_score < 40:
        risk_mode = "RISK-OFF"
        risk_color = "#FF1744"
        risk_message = "Piyasa RISK-OFF modunda. Güvenli limanlara (Nakit/Altın) geçiş mantıklı görünüyor."
        risk_emoji = "🔴"
    else:
        risk_mode = "NÖTR"
        risk_color = "#FF9800"
        risk_message = "Piyasa karışık sinyaller veriyor. Dikkatli olun ve pozisyon boyutunu küçük tutun."
        risk_emoji = "🟡"
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
        <div style="text-align: center; padding: 25px; background: linear-gradient(135deg, {risk_color}22, {risk_color}44); border-radius: 15px; border: 3px solid {risk_color};">
            <h1 style="color: {risk_color}; margin: 0; font-size: 3.5rem;">{risk_score}</h1>
            <h3 style="color: {risk_color}; margin: 5px 0;">{risk_emoji} {risk_mode}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if risk_score > 70:
            st.success(risk_message)
        elif risk_score < 40:
            st.error(risk_message)
        else:
            st.warning(risk_message)
        
        # Faktör detayları
        with st.expander("📋 Skor Faktörleri"):
            for factor, detail in risk_factors:
                st.write(f"**{factor}**: {detail}")
    
    st.divider()
    
    # ==================== MAKRO METRİKLER ====================
    st.subheader("🌍 Küresel Göstergeler")
    
    # İlk satır: Para & Tahvil
    cols = st.columns(4)
    
    with cols[0]:
        if macro_data.get('DXY'):
            dxy = macro_data['DXY']
            st.metric("💵 DXY (Dolar)", f"{dxy['value']:.2f}", f"{dxy['change']:+.2f}%")
        else:
            st.metric("💵 DXY", "—")
    
    with cols[1]:
        if macro_data.get('US10Y'):
            bonds = macro_data['US10Y']
            st.metric("📜 ABD 10Y Tahvil", f"%{bonds['value']:.2f}", f"{bonds['change']:+.2f}%")
        else:
            st.metric("📜 ABD 10Y", "—")
    
    with cols[2]:
        if macro_data.get('VIX'):
            vix = macro_data['VIX']
            vix_status = "🟢" if vix['value'] < 20 else "🟡" if vix['value'] < 30 else "🔴"
            st.metric(f"😱 VIX {vix_status}", f"{vix['value']:.1f}", f"{vix['change']:+.2f}%")
        else:
            st.metric("😱 VIX", "—")
    
    with cols[3]:
        if macro_data.get('USDJPY'):
            jpy = macro_data['USDJPY']
            st.metric("🇯🇵 USD/JPY", f"{jpy['value']:.2f}", f"{jpy['change']:+.2f}%")
        else:
            st.metric("🇯🇵 USD/JPY", "—")
    
    # İkinci satır: Emtia
    cols2 = st.columns(4)
    
    with cols2[0]:
        if macro_data.get('Gold'):
            gold = macro_data['Gold']
            st.metric("🥇 Altın", f"${gold['value']:,.0f}", f"{gold['change']:+.2f}%")
        else:
            st.metric("🥇 Altın", "—")
    
    with cols2[1]:
        if macro_data.get('Silver'):
            silver = macro_data['Silver']
            st.metric("🥈 Gümüş", f"${silver['value']:.2f}", f"{silver['change']:+.2f}%")
        else:
            st.metric("🥈 Gümüş", "—")
    
    with cols2[2]:
        if macro_data.get('Oil'):
            oil = macro_data['Oil']
            st.metric("�️ WTI Petrol", f"${oil['value']:.2f}", f"{oil['change']:+.2f}%")
        else:
            st.metric("🛢️ WTI Petrol", "—")
    
    with cols2[3]:
        # Gold/Silver oranı
        if macro_data.get('Gold') and macro_data.get('Silver'):
            gold_val = macro_data['Gold']['value']
            silver_val = macro_data['Silver']['value']
            ratio = gold_val / silver_val if silver_val > 0 else 0
            ratio_status = "🟢 Ucuz" if ratio > 80 else "🔴 Pahalı" if ratio < 60 else "🟡"
            st.metric(f"Au/Ag {ratio_status}", f"{ratio:.1f}x")
        else:
            st.metric("Au/Ag Oranı", "—")
    
    st.divider()
    
    # ==================== KORELASYON ISIL HARİTASI ====================
    st.subheader("🔥 Korelasyon Isı Haritası")
    st.caption("Son 30 gün - BTC, ETH, DXY, VIX, Gold, Oil, JPY, S&P500")
    
    with st.spinner("Korelasyon hesaplanıyor..."):
        corr_matrix, corr_error = fetch_correlation_heatmap_data(30)
    
    if corr_matrix is not None:
        import plotly.express as px
        
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1,
            aspect='auto'
        )
        
        fig.update_layout(
            template="plotly_dark",
            height=400,
            margin=dict(l=0, r=0, t=30, b=0),
            coloraxis_colorbar=dict(title="r")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📊 Korelasyon Yorumu"):
            st.write("• **BTC-DXY**: Negatif = zayıf dolar BTC'ye olumlu")
            st.write("• **BTC-VIX**: Korku artınca BTC genellikle düşer")
            st.write("• **Gold-DXY**: Genellikle negatif korelasyon")
    else:
        st.warning(f"Korelasyon verisi alınamadı: {corr_error}")
    
    st.divider()
    
    # ==================== LİKİDİTE vs BTC ====================
    st.subheader("💰 Likidite vs Bitcoin")
    st.caption("TLT (Uzun vadeli tahvil ETF) likidite proxy'si olarak kullanılır")
    
    if liquidity_data and liquidity_data.get('btc_history') is not None:
        tlt_hist = liquidity_data['tlt_history']
        btc_hist = liquidity_data['btc_history']
        
        fig = go.Figure()
        
        # TLT (sol eksen)
        fig.add_trace(go.Scatter(
            x=tlt_hist.index,
            y=tlt_hist['Close'],
            name='TLT (Likidite)',
            line=dict(color='#2196F3', width=2),
            yaxis='y'
        ))
        
        # BTC (sağ eksen)
        fig.add_trace(go.Scatter(
            x=btc_hist.index,
            y=btc_hist['Close'],
            name='Bitcoin',
            line=dict(color='#FF9800', width=2),
            yaxis='y2'
        ))
        
        fig.update_layout(
            template="plotly_dark",
            height=350,
            margin=dict(l=0, r=0, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            yaxis=dict(title="TLT ($)", side="left"),
            yaxis2=dict(title="BTC ($)", side="right", overlaying="y")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Likidite açıklaması
        with st.expander("💡 Likidite Neden Önemli?"):
            st.write("""
            **TLT yükselirse** → Tahvil faizleri düşüyor → Fed gevşiyor → Likidite artıyor → BTC için olumlu
            
            **TLT düşerse** → Tahvil faizleri yükseliyor → Fed sıkılaştırıyor → Likidite azalıyor → BTC için olumsuz
            
            Bu ilişki %100 değildir ama uzun vadeli trendlerde genellikle geçerlidir.
            """)
    else:
        st.warning("Likidite karşılaştırma verisi alınamadı")
    
    st.divider()
    
    # ==================== GETİRİ EĞRİSİ ====================
    st.subheader("📉 Getiri Eğrisi (10Y - 2Y Spread)")
    
    if yield_data:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            spread = yield_data['spread']
            inverted = yield_data['inverted']
            
            if inverted:
                spread_status = "🔴 TERS"
                spread_color = "#FF1744"
            elif spread < 0.5:
                spread_status = "🟡 DÜZLEŞEN"
                spread_color = "#FF9800"
            else:
                spread_status = "🟢 NORMAL"
                spread_color = "#00C853"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; background: {spread_color}22; border-radius: 10px; border: 2px solid {spread_color};">
                <h2 style="color: {spread_color}; margin: 0;">{spread:.2f}%</h2>
                <p style="color: {spread_color}; margin: 0;">{spread_status}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric("📈 10Y Getiri", f"%{yield_data['us10y']:.2f}")
        
        with col3:
            st.metric("📊 2Y Getiri", f"%{yield_data['us02y']:.2f}")
        
        # Spread geçmişi grafiği
        if yield_data.get('history'):
            import pandas as pd
            spread_df = pd.DataFrame(yield_data['history'])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=spread_df['date'],
                y=spread_df['spread'],
                mode='lines',
                fill='tozeroy',
                line=dict(color='#4CAF50' if not inverted else '#FF1744', width=2),
                name='10Y-2Y Spread'
            ))
            
            # Sıfır çizgisi
            fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Inversiyon")
            
            fig.update_layout(
                template="plotly_dark",
                height=250,
                margin=dict(l=0, r=0, t=20, b=20),
                yaxis_title="Spread (%)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("🚨 Resesyon Alarmı Nedir?"):
            st.write("""
            **Getiri eğrisi** uzun vadeli faizler (10Y) ile kısa vadeli faizler (2Y) arasındaki farktır.
            
            **Normal eğri (pozitif spread)**: Uzun vade > Kısa vade → Ekonomi sağlıklı
            
            **Ters eğri (negatif spread)**: Uzun vade < Kısa vade → **Resesyon sinyali**
            
            Tarihsel olarak, ters getiri eğrisi 6-18 ay içinde resesyonu önceden tahmin etmiştir.
            """)
    else:
        st.warning(f"Getiri eğrisi verisi alınamadı: {yield_err}")
    
    st.divider()
    
    # ==================== KREDİ RİSKİ ====================
    st.subheader("💳 Kredi Riski ve Ekonomik Sağlık")
    
    if credit_data:
        credit_cols = st.columns(3)
        
        # Kredi Spreadi
        with credit_cols[0]:
            if credit_data.get('credit'):
                cr = credit_data['credit']
                cr_color = "#FF1744" if cr['risk_level'] == "YÜKSEK" else "#00C853" if cr['risk_level'] == "DÜŞÜK" else "#FF9800"
                st.markdown(f"""
                <div style="text-align: center; padding: 15px; background: {cr_color}22; border-radius: 10px; border: 2px solid {cr_color};">
                    <p style="margin: 0; color: #888;">HY/IG Spread</p>
                    <h3 style="color: {cr_color}; margin: 5px 0;">{cr['risk_level']}</h3>
                    <p style="color: {cr_color}; margin: 0;">{cr['change_30d']:+.1f}% (30g)</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Reel Faiz
        with credit_cols[1]:
            if credit_data.get('real_yield'):
                ry = credit_data['real_yield']
                ry_color = "#00C853" if ry['trend'] == "DÜŞÜYOR" else "#FF1744" if ry['trend'] == "YÜKSELIYOR" else "#FF9800"
                st.markdown(f"""
                <div style="text-align: center; padding: 15px; background: {ry_color}22; border-radius: 10px; border: 2px solid {ry_color};">
                    <p style="margin: 0; color: #888;">Reel Faiz</p>
                    <h3 style="color: {ry_color}; margin: 5px 0;">{ry['trend']}</h3>
                    <p style="color: {ry_color}; margin: 0;">TIP: {ry['change_30d']:+.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Copper/Gold
        with credit_cols[2]:
            if credit_data.get('copper_gold'):
                cg = credit_data['copper_gold']
                cg_color = "#00C853" if cg['outlook'] == "İYİMSER" else "#FF1744" if cg['outlook'] == "KÖTÜMSER" else "#FF9800"
                st.markdown(f"""
                <div style="text-align: center; padding: 15px; background: {cg_color}22; border-radius: 10px; border: 2px solid {cg_color};">
                    <p style="margin: 0; color: #888;">Cu/Au Oranı</p>
                    <h3 style="color: {cg_color}; margin: 5px 0;">{cg['outlook']}</h3>
                    <p style="color: {cg_color}; margin: 0;">{cg['change_30d']:+.1f}% (30g)</p>
                </div>
                """, unsafe_allow_html=True)
        
        with st.expander("💡 Göstergeler Ne Anlama Geliyor?"):
            st.write("""
            **HY/IG Spread**: High Yield vs Investment Grade tahvil oranı. Düşüyorsa → Kredi riski artıyor
            
            **Reel Faiz**: TIP ETF ile ölçülür. Düşüyorsa → BTC ve Altın lehine
            
            **Cu/Au Oranı**: Bakır/Altın oranı ekonomik sağlık göstergesi. Yükseliyorsa → Ekonomik iyimserlik
            """)
    else:
        st.warning(f"Kredi verisi alınamadı: {credit_err}")
    
    st.divider()
    
    # ==================== BTC KARAKTERİ ====================
    st.subheader("🎭 BTC Karakteri: Teknoloji mi, Dijital Altın mı?")
    
    if correlation_data:
        char_cols = st.columns([1, 2])
        
        with char_cols[0]:
            char_color = "#2196F3" if "Teknoloji" in correlation_data['btc_character'] else "#FFD700" if "Altın" in correlation_data['btc_character'] else "#9C27B0"
            st.markdown(f"""
            <div style="text-align: center; padding: 25px; background: {char_color}22; border-radius: 15px; border: 3px solid {char_color};">
                <h2 style="color: {char_color}; margin: 0;">{correlation_data['btc_character']}</h2>
                <p style="color: #888; margin: 10px 0;">{correlation_data['character_detail']}</p>
                <p style="margin: 5px 0;">Nasdaq: <b>{correlation_data['last_nasdaq_corr']:.2f}</b></p>
                <p style="margin: 5px 0;">Gold: <b>{correlation_data['last_gold_corr']:.2f}</b></p>
            </div>
            """, unsafe_allow_html=True)
        
        with char_cols[1]:
            # Rolling correlation grafiği
            if correlation_data.get('history') and correlation_data['history'].get('dates'):
                corr_hist = correlation_data['history']
                
                fig = go.Figure()
                
                if corr_hist.get('btc_nasdaq'):
                    fig.add_trace(go.Scatter(
                        x=corr_hist['dates'],
                        y=corr_hist['btc_nasdaq'],
                        name='BTC-Nasdaq',
                        line=dict(color='#2196F3', width=2)
                    ))
                
                if corr_hist.get('btc_gold'):
                    fig.add_trace(go.Scatter(
                        x=corr_hist['dates'],
                        y=corr_hist['btc_gold'],
                        name='BTC-Gold',
                        line=dict(color='#FFD700', width=2)
                    ))
                
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                
                fig.update_layout(
                    template="plotly_dark",
                    height=250,
                    margin=dict(l=0, r=0, t=20, b=20),
                    yaxis_title="Korelasyon",
                    yaxis=dict(range=[-1, 1]),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02)
                )
                
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"Korelasyon verisi alınamadı: {corr_err}")
    
    st.divider()
    
    # ==================== JEOPOLİTİK VE TİCARET ====================
    st.subheader("🌐 Jeopolitik Risk ve Küresel Ticaret")
    
    if geo_data:
        geo_cols = st.columns(4)
        
        # GPR (Jeopolitik Risk)
        with geo_cols[0]:
            if geo_data.get('gpr'):
                gpr = geo_data['gpr']
                gpr_color = "#FF1744" if gpr['level'] == "YÜKSEK" else "#00C853" if gpr['level'] == "DÜŞÜK" else "#FF9800"
                st.markdown(f"""
                <div style="text-align: center; padding: 15px; background: {gpr_color}22; border-radius: 10px; border: 2px solid {gpr_color};">
                    <p style="margin: 0; color: #888;">🎯 Jeopolitik Risk</p>
                    <h2 style="color: {gpr_color}; margin: 5px 0;">{gpr['score']:.0f}</h2>
                    <p style="color: {gpr_color}; margin: 0;">{gpr['level']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Baltic Dry Index
        with geo_cols[1]:
            if geo_data.get('trade'):
                bdi = geo_data['trade']
                bdi_color = "#00C853" if bdi['outlook'] == "CANLI" else "#FF1744" if bdi['outlook'] == "DURGUN" else "#FF9800"
                st.markdown(f"""
                <div style="text-align: center; padding: 15px; background: {bdi_color}22; border-radius: 10px; border: 2px solid {bdi_color};">
                    <p style="margin: 0; color: #888;">🚢 Küresel Ticaret</p>
                    <h3 style="color: {bdi_color}; margin: 5px 0;">{bdi['outlook']}</h3>
                    <p style="color: {bdi_color}; margin: 0;">{bdi['change_30d']:+.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Bank Stress
        with geo_cols[2]:
            if geo_data.get('bank'):
                bank = geo_data['bank']
                bank_color = "#FF1744" if bank['stress_level'] == "YÜKSEK" else "#00C853" if bank['stress_level'] == "DÜŞÜK" else "#FF9800"
                st.markdown(f"""
                <div style="text-align: center; padding: 15px; background: {bank_color}22; border-radius: 10px; border: 2px solid {bank_color};">
                    <p style="margin: 0; color: #888;">🏦 Banka Stresi</p>
                    <h3 style="color: {bank_color}; margin: 5px 0;">{bank['stress_level']}</h3>
                    <p style="color: {bank_color}; margin: 0;">{bank['change_30d']:+.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Asset Rotation
        with geo_cols[3]:
            if geo_data.get('ratios') and geo_data['ratios'].get('nasdaq_gold'):
                rot = geo_data['ratios']['nasdaq_gold']
                rot_color = "#00C853" if rot['rotation'] == "RISK-ON" else "#FF1744" if rot['rotation'] == "RISK-OFF" else "#FF9800"
                st.markdown(f"""
                <div style="text-align: center; padding: 15px; background: {rot_color}22; border-radius: 10px; border: 2px solid {rot_color};">
                    <p style="margin: 0; color: #888;">🔄 Varlık Rotasyonu</p>
                    <h3 style="color: {rot_color}; margin: 5px 0;">{rot['rotation']}</h3>
                    <p style="color: {rot_color}; margin: 0;">NQ/Au: {rot['change_30d']:+.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
        
        with st.expander("💡 Göstergeler Hakkında"):
            st.write("""
            **Jeopolitik Risk (GPR)**: VIX + Altın volatilitesi bazlı proxy. Yüksekse küresel belirsizlik var.
            
            **Küresel Ticaret (BDI)**: Baltic Dry Index - nakliye maliyetleri. Yükseliyorsa ticaret canlı.
            
            **Banka Stresi**: KBE/TLT oranı. Düşüyorsa bankalar stres altında.
            
            **Varlık Rotasyonu**: Nasdaq/Altın oranı. Yükseliyorsa risk-on, düşüyorsa risk-off.
            """)
    else:
        st.warning(f"Jeopolitik veri alınamadı: {geo_err}")
    
    st.divider()
    
    # ==================== MASTER FEATURES ====================
    with st.expander("🤖 XGBoost Feature Matrix (ML Ready)"):
        if master_features:
            st.json(master_features)
            st.success(f"✅ {len(master_features)} feature hazır. st.session_state['master_features_final'] içinde kaydedildi.")
        else:
            st.warning("Feature matrix henüz hazır değil.")


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
