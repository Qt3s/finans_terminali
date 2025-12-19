"""
Profesyonel Finans Terminali v2.0
Tüm modülleri (Mikabot, AI, Makro) tek profesyonel çatı altında toplayan
modüler, yüksek performanslı Streamlit terminali.

Özellikler:
- 🏠 KOKPİT: Executive Summary, kritik metrikler
- 📡 PİYASA RADARI: TrendString, InOut, SVI, Orderbook
- 🧠 QUANT LAB: XGBoost, SHAP, FFT Döngü, Kelly
- 🌍 MAKRO & TEMEL: DXY, Faizler, On-Chain, Sentiment
- ⚙️ SİSTEM: Backtest, Ayarlar
"""

# ==================== IMPORTS ====================
# Core Libraries
import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

# Visualization
import plotly.graph_objects as go
import plotly.express as px

# Data Sources
import requests
import ccxt
import yfinance as yf

# Machine Learning
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Technical Analysis
from scipy.signal import argrelextrema
from scipy.fft import fft, fftfreq  # FFT Döngü Analizi için

# Blockchain (optional)
try:
    from web3 import Web3
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False



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

CRYPTO_SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT", "AVAX/USDT", "POL/USDT"]
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


# ==================== VERİ TEMİZLİĞİ UTILITIES ====================

def clean_dataframe(df, method='ffill_interpolate'):
    """
    DataFrame'deki NaN ve inf değerlerini temizler.
    
    Args:
        df: Temizlenecek DataFrame
        method: 'ffill', 'interpolate', veya 'ffill_interpolate'
    
    Returns:
        Temizlenmiş DataFrame
    """
    
    df = df.copy()
    
    # Inf değerleri NaN'a çevir
    df = df.replace([np.inf, -np.inf], np.nan)
    
    if method == 'ffill':
        df = df.ffill().bfill()
    elif method == 'interpolate':
        df = df.interpolate(method='linear').ffill().bfill()
    elif method == 'ffill_interpolate':
        # Önce forward fill, sonra interpolasyon
        df = df.ffill()
        df = df.interpolate(method='linear')
        df = df.bfill()  # Başlangıç NaN'ları için
    
    return df


def apply_median_filter(series, window: int = 5, threshold: float = 3.0):
    """
    Outlier/spike tespiti ve düzeltmesi için medyan filtre.
    
    Args:
        series: Pandas Series
        window: Medyan pencere boyutu
        threshold: Standart sapma eşiği (3 = %99.7 güven)
    
    Returns:
        Filtrelenmiş Series
    """
    
    series = series.copy()
    
    # Rolling medyan ve std
    rolling_median = series.rolling(window=window, center=True, min_periods=1).median()
    rolling_std = series.rolling(window=window, center=True, min_periods=1).std()
    
    # Outlier tespiti
    diff = np.abs(series - rolling_median)
    outliers = diff > (threshold * rolling_std)
    
    # Outlier'ları medyan ile değiştir
    series[outliers] = rolling_median[outliers]
    
    return series


def merge_time_series(dfs: list, how: str = 'outer', fill_method: str = 'ffill_interpolate'):
    """
    Farklı zaman serilerini birleştirir ve hizalar.
    
    Args:
        dfs: DataFrame listesi (her biri DatetimeIndex olmalı)
        how: 'inner' veya 'outer' merge
        fill_method: NaN doldurma metodu
    
    Returns:
        Birleştirilmiş DataFrame
    """
    
    if not dfs:
        return pd.DataFrame()
    
    # İlk DataFrame ile başla
    result = dfs[0].copy()
    
    # Diğerlerini birleştir
    for df in dfs[1:]:
        result = result.join(df, how=how, rsuffix='_dup')
        
        # Duplicate sütunları kaldır
        result = result.loc[:, ~result.columns.str.endswith('_dup')]
    
    # Temizle
    result = clean_dataframe(result, method=fill_method)
    
    return result


# ==================== VERİ ÇEKİCİ FONKSİYONLAR ====================

def get_exchange_instance(config):
    """Borsa instance'ı oluşturur."""
    exchange_class = getattr(ccxt, config['class'])
    return exchange_class(config['options'])


@st.cache_data(ttl=120, show_spinner=False)  # Fiyat verileri: 2 dakika
def fetch_crypto_ticker(symbol: str):
    """Kripto fiyat bilgisi (fallback mekanizması)."""
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


@st.cache_data(ttl=120, show_spinner=False)  # Fiyat verileri: 2 dakika
def fetch_crypto_ohlcv(symbol: str, timeframe: str, limit: int = 200):
    """Kripto OHLCV verisi + EMA hesaplama."""
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


# ==================== MIKABOT-STYLE ANALİZ MODÜLLER ====================

@st.cache_data(ttl=300, show_spinner=False)
def calculate_altpower_score():
    """
    Binance üzerinden BTC ve 20 majör altcoinin 24H performansını karşılaştırır.
    BTC'yi geçen altcoin oranını hesaplar.
    
    Returns:
        tuple: (altpower_score: float, btc_change: float)
        - altpower_score: 0-100 arası skor (BTC'yi geçen altcoin %)
        - btc_change: BTC'nin 24H değişimi
    """
    
    ALTCOINS = [
        'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT',
        'DOGE/USDT', 'AVAX/USDT', 'TRX/USDT', 'DOT/USDT', 'POL/USDT',
        'LTC/USDT', 'LINK/USDT', 'UNI/USDT', 'ATOM/USDT', 'ETC/USDT',
        'FIL/USDT', 'NEAR/USDT', 'AAVE/USDT', 'QNT/USDT', 'ALGO/USDT'
    ]
    
    try:
        exchange = ccxt.kucoin({'enableRateLimit': True})
        
        # BTC 24H değişimini al
        btc_ticker = exchange.fetch_ticker('BTC/USDT')
        btc_change = btc_ticker.get('percentage', 0) or 0
        
        # Altcoinlerin kaçı BTC'den iyi performans gösteriyor
        outperforming = 0
        
        for symbol in ALTCOINS:
            try:
                ticker = exchange.fetch_ticker(symbol)
                alt_change = ticker.get('percentage', 0) or 0
                if alt_change > btc_change:
                    outperforming += 1
            except:
                continue
        
        # Skor: (BTC'yi geçen sayısı / 20) * 100
        altpower_score = (outperforming / 20) * 100
        
        return altpower_score, btc_change
        
    except Exception as e:
        # Hata durumunda varsayılan değerler
        return 50.0, 0.0


@st.cache_data(ttl=600, show_spinner=False)
def calculate_altpower():
    """
    Top 50 altcoinin BTC paritesindeki 24H değişimlerini analiz eder.
    Pozitif ayrışanların yüzdesini hesaplar.
    
    Returns:
        dict: altpower_score (0-100), positive_count, total_count, details (top 10)
    """
    
    try:
        exchange = ccxt.kucoin({'enableRateLimit': True})
        markets = exchange.load_markets()
        
        # BTC pariteli altcoinleri filtrele (ilk 50)
        btc_pairs = [s for s in markets if s.endswith('/BTC')][:50]
        
        positive_count = 0
        total_count = 0
        details = []
        
        for symbol in btc_pairs:
            try:
                ticker = exchange.fetch_ticker(symbol)
                change_24h = ticker.get('percentage', 0) or 0
                details.append({'symbol': symbol.split('/')[0], 'change': change_24h})
                if change_24h > 0:
                    positive_count += 1
                total_count += 1
                time.sleep(0.5)  # Rate limit önleme
            except:
                continue
        
        altpower_score = (positive_count / total_count * 100) if total_count > 0 else 50
        
        return {
            'altpower_score': altpower_score,
            'positive_count': positive_count,
            'total_count': total_count,
            'details': sorted(details, key=lambda x: x['change'], reverse=True)[:10]
        }
    except Exception as e:
        return {
            'altpower_score': 50,
            'positive_count': 0,
            'total_count': 0,
            'details': [],
            'error': str(e)
        }


@st.cache_data(ttl=600, show_spinner=False)
def calculate_inout_flow():
    """
    10 majör coin için son 1 saatlik alış/satış hacim dengesini hesaplar.
    
    Returns:
        list: Her coin için symbol, buy_volume, sell_volume, net_flow, flow_pct, flow_type
    """
    
    MAJOR_COINS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
                   'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'DOT/USDT', 'POL/USDT']
    
    try:
        exchange = ccxt.kucoin({'enableRateLimit': True})
        results = []
        
        for symbol in MAJOR_COINS:
            try:
                # Son 1 saatlik mumları çek (60 dakika = 60 x 1m mumlar)
                ohlcv = exchange.fetch_ohlcv(symbol, '1m', limit=60)
                
                buy_volume = 0
                sell_volume = 0
                
                for candle in ohlcv:
                    open_p, high, low, close, volume = candle[1:6]
                    # Kapanış >= Açılış = Alış baskın
                    if close >= open_p:
                        buy_volume += volume
                    else:
                        sell_volume += volume
                
                net_flow = buy_volume - sell_volume
                total_volume = buy_volume + sell_volume
                flow_pct = (net_flow / total_volume * 100) if total_volume > 0 else 0
                
                results.append({
                    'symbol': symbol.split('/')[0],
                    'buy_volume': buy_volume,
                    'sell_volume': sell_volume,
                    'net_flow': net_flow,
                    'flow_pct': flow_pct,
                    'flow_type': 'BUY' if net_flow > 0 else 'SELL'
                })
                
                time.sleep(0.5)  # Rate limit önleme
            except Exception:
                results.append({
                    'symbol': symbol.split('/')[0],
                    'buy_volume': 0,
                    'sell_volume': 0,
                    'net_flow': 0,
                    'flow_pct': 0,
                    'flow_type': 'N/A'
                })
        
        return results
    except Exception as e:
        return []


@st.cache_data(ttl=600, show_spinner=False)
def calculate_trendstring(symbol: str = 'BTC/USDT'):
    """
    Son 5 adet 4H mumun kapanış yönünü +/- olarak gösterir.
    
    Args:
        symbol: Kripto para sembolü (default: BTC/USDT)
    
    Returns:
        dict: trendstring (+/-), visual (emoji), bullish_count
    """
    
    try:
        exchange = ccxt.kucoin({'enableRateLimit': True})
        ohlcv = exchange.fetch_ohlcv(symbol, '4h', limit=6)  # 6 çek, 5 karşılaştır
        
        if len(ohlcv) < 6:
            return {'trendstring': '?????', 'visual': '❓❓❓❓❓', 'bullish_count': 0}
        
        trend_chars = []
        visual_chars = []
        bullish_count = 0
        
        for i in range(1, 6):  # Son 5 mum
            prev_close = ohlcv[i-1][4]
            curr_close = ohlcv[i][4]
            
            if curr_close >= prev_close:
                trend_chars.append('+')
                visual_chars.append('📈')
                bullish_count += 1
            else:
                trend_chars.append('-')
                visual_chars.append('📉')
        
        return {
            'trendstring': ''.join(trend_chars),
            'visual': ''.join(visual_chars),
            'bullish_count': bullish_count
        }
    except Exception as e:
        return {'trendstring': '?????', 'visual': '❓❓❓❓❓', 'bullish_count': 0, 'error': str(e)}


@st.cache_data(ttl=600, show_spinner=False)
def fetch_market_radar_data():
    """
    Top 10 majör coin için Piyasa Radarı verisi.
    TrendString (4H mum), InOut momentum skoru ve fiyat bilgisi.
    
    Returns:
        list: Her coin için radar verisi (symbol, price, trend, inout, change)
    """
    
    TOP_COINS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT',
                 'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'DOT/USDT', 'POL/USDT']
    
    try:
        exchange = ccxt.kucoin({'enableRateLimit': True})
        results = []
        
        # Önce tüm tickerları çek (hacim ortalaması için)
        all_volumes = []
        tickers_cache = {}
        
        for symbol in TOP_COINS:
            try:
                ticker = exchange.fetch_ticker(symbol)
                tickers_cache[symbol] = ticker
                quote_vol = ticker.get('quoteVolume', 0) or 0
                all_volumes.append(quote_vol)
            except:
                continue
        
        avg_volume = sum(all_volumes) / len(all_volumes) if all_volumes else 1
        
        for symbol in TOP_COINS:
            try:
                ticker = tickers_cache.get(symbol)
                if not ticker:
                    ticker = exchange.fetch_ticker(symbol)
                
                price = ticker.get('last', 0) or 0
                change_24h = ticker.get('percentage', 0) or 0
                quote_volume = ticker.get('quoteVolume', 0) or 0
                
                # ===== TRENDSTRING: Son 5 adet 4H mum =====
                ohlcv = exchange.fetch_ohlcv(symbol, '4h', limit=5)
                trend_chars = []
                trend_html = []
                
                for candle in ohlcv:
                    open_p, high, low, close = candle[1:5]
                    if close >= open_p:
                        trend_chars.append('+')
                        trend_html.append('<span style="color:#00C853;">+</span>')
                    else:
                        trend_chars.append('-')
                        trend_html.append('<span style="color:#FF1744;">-</span>')
                
                trendstring = ''.join(trend_chars)
                trend_colored = ''.join(trend_html)
                
                # ===== INOUT MOMENTUM SKORU =====
                # Skor = (Fiyat Değişimi %) × (Hacim / Ortalama Hacim)
                volume_ratio = quote_volume / avg_volume if avg_volume > 0 else 1
                inout_score = change_24h * volume_ratio
                
                # InOut durumu belirleme
                if inout_score > 5:
                    inout_status = "🟢 Güçlü Giriş"
                elif inout_score > 1:
                    inout_status = "🟢 Giriş"
                elif inout_score < -5:
                    inout_status = "🔴 Güçlü Çıkış"
                elif inout_score < -1:
                    inout_status = "🔴 Çıkış"
                else:
                    inout_status = "⚪ Nötr"
                
                results.append({
                    'Coin': symbol.split('/')[0],
                    'Fiyat': price,
                    'TrendString': trendstring,
                    'TrendHTML': trend_colored,
                    'InOut': inout_status,
                    'InOutScore': inout_score,
                    '24s Değişim': change_24h
                })
                
                time.sleep(0.3)  # Rate limit önleme
                
            except Exception as e:
                results.append({
                    'Coin': symbol.split('/')[0],
                    'Fiyat': 0,
                    'TrendString': '?????',
                    'TrendHTML': '?????',
                    'InOut': '❓ Veri Yok',
                    'InOutScore': 0,
                    '24s Değişim': 0
                })
        
        return results
        
    except Exception as e:
        return []


# ==================== DERİN ANALİZ MODÜLLER ====================

@st.cache_data(ttl=600, show_spinner=False)
def calculate_squeeze_volatility():
    """
    SVI (Squeeze Volatility Index) - Bollinger Band sıkışma tespiti.
    Bandwidth küçük = Fiyat patlayabilir.
    
    Returns:
        list: Her coin için sıkışma durumu
    """
    
    TOP_COINS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT',
                 'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'DOT/USDT', 'POL/USDT']
    
    SQUEEZE_THRESHOLD = 0.04  # %4'ün altı sıkışma
    
    try:
        exchange = ccxt.kucoin({'enableRateLimit': True})
        results = []
        
        for symbol in TOP_COINS:
            try:
                # Son 20 mum (Bollinger için standart)
                ohlcv = exchange.fetch_ohlcv(symbol, '4h', limit=20)
                closes = np.array([c[4] for c in ohlcv])
                
                # Bollinger Bantları
                sma = np.mean(closes)
                std = np.std(closes)
                upper = sma + (2 * std)
                lower = sma - (2 * std)
                
                # Bandwidth hesaplama
                bandwidth = (upper - lower) / sma if sma > 0 else 0
                
                # Sıkışma durumu
                if bandwidth < SQUEEZE_THRESHOLD:
                    squeeze_status = "🔥 Sıkışıyor"
                    squeeze_alert = True
                elif bandwidth < SQUEEZE_THRESHOLD * 1.5:
                    squeeze_status = "⚠️ Dikkat"
                    squeeze_alert = False
                else:
                    squeeze_status = "✅ Normal"
                    squeeze_alert = False
                
                results.append({
                    'Coin': symbol.split('/')[0],
                    'Bandwidth': bandwidth * 100,  # Yüzde olarak
                    'SqueezeStatus': squeeze_status,
                    'SqueezeAlert': squeeze_alert,
                    'Price': closes[-1] if len(closes) > 0 else 0
                })
                
            except:
                results.append({
                    'Coin': symbol.split('/')[0],
                    'Bandwidth': 0,
                    'SqueezeStatus': '❓ Veri Yok',
                    'SqueezeAlert': False,
                    'Price': 0
                })
        
        return results
        
    except Exception as e:
        return []


@st.cache_data(ttl=1800, show_spinner=False)  # 30 dakika cache
def fetch_correlation_matrix():
    """
    Son 30 günlük fiyat korelasyonu matrisi.
    
    Returns:
        tuple: (correlation_matrix, coin_list)
    """
    
    COINS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'BNB-USD',
             'ADA-USD', 'DOGE-USD', 'AVAX-USD', 'DOT-USD', 'MATIC-USD']
    
    try:
        # Tüm coinlerin 30 günlük kapanış fiyatlarını çek
        closes_dict = {}
        
        for coin in COINS:
            try:
                ticker = yf.Ticker(coin)
                hist = ticker.history(period='30d')
                if not hist.empty:
                    closes_dict[coin.replace('-USD', '')] = hist['Close'].values
            except:
                continue
        
        if len(closes_dict) < 3:
            return None, []
        
        # DataFrame oluştur ve korelasyon hesapla
        df = pd.DataFrame(closes_dict)
        
        # Eksik günleri doldur
        df = df.ffill().bfill()
        
        # Korelasyon matrisi
        corr_matrix = df.corr()
        
        return corr_matrix, list(closes_dict.keys())
        
    except Exception as e:
        return None, []


@st.cache_data(ttl=600, show_spinner=False)
def calculate_smart_scores():
    """
    Smart Score - Her coin için tek kalite puanı.
    
    Formül: (Trend * 0.4) + (Hacim * 0.4) + (Volatilite * 0.2)
    
    Returns:
        list: Her coin için Smart Score (0-100)
    """
    
    TOP_COINS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT',
                 'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'DOT/USDT', 'POL/USDT']
    
    try:
        exchange = ccxt.kucoin({'enableRateLimit': True})
        results = []
        
        # Ortalama hacim için tüm verileri topla
        all_volumes = []
        all_data = {}
        
        for symbol in TOP_COINS:
            try:
                ticker = exchange.fetch_ticker(symbol)
                ohlcv = exchange.fetch_ohlcv(symbol, '4h', limit=14)  # RSI için 14 periyot
                
                all_data[symbol] = {
                    'ticker': ticker,
                    'ohlcv': ohlcv
                }
                all_volumes.append(ticker.get('quoteVolume', 0) or 0)
            except:
                continue
        
        avg_volume = np.mean(all_volumes) if all_volumes else 1
        
        for symbol in TOP_COINS:
            try:
                data = all_data.get(symbol)
                if not data:
                    continue
                
                ticker = data['ticker']
                ohlcv = data['ohlcv']
                closes = np.array([c[4] for c in ohlcv])
                
                # ===== TREND PUANI (0-100) =====
                # RSI hesaplama
                if len(closes) >= 14:
                    deltas = np.diff(closes)
                    gains = np.where(deltas > 0, deltas, 0)
                    losses = np.where(deltas < 0, -deltas, 0)
                    avg_gain = np.mean(gains[-14:])
                    avg_loss = np.mean(losses[-14:])
                    rs = avg_gain / avg_loss if avg_loss > 0 else 100
                    rsi = 100 - (100 / (1 + rs))
                else:
                    rsi = 50
                
                # EMA durumu (fiyat EMA üstünde mi?)
                ema_20 = np.mean(closes[-20:]) if len(closes) >= 20 else np.mean(closes)
                price = closes[-1] if len(closes) > 0 else 0
                ema_bonus = 20 if price > ema_20 else 0
                
                # RSI'ı 0-80 aralığına normalize et, EMA bonus ekle
                trend_score = min(100, max(0, (rsi * 0.8) + ema_bonus))
                
                # ===== HACİM PUANI (0-100) =====
                quote_volume = ticker.get('quoteVolume', 0) or 0
                volume_ratio = quote_volume / avg_volume if avg_volume > 0 else 1
                volume_score = min(100, volume_ratio * 50)  # 2x ortalama = 100 puan
                
                # ===== VOLATİLİTE PUANI (0-100) =====
                # Düşük volatilite = sıkışma = yüksek puan
                if len(closes) >= 20:
                    std = np.std(closes[-20:])
                    mean = np.mean(closes[-20:])
                    bandwidth = (std * 2) / mean if mean > 0 else 0
                    # Düşük bandwidth = yüksek puan
                    volatility_score = max(0, 100 - (bandwidth * 1000))
                else:
                    volatility_score = 50
                
                # ===== SMART SCORE =====
                smart_score = (trend_score * 0.4) + (volume_score * 0.4) + (volatility_score * 0.2)
                smart_score = min(100, max(0, smart_score))
                
                # Grade belirleme
                if smart_score >= 75:
                    grade = "🟢 A"
                elif smart_score >= 60:
                    grade = "🟡 B"
                elif smart_score >= 40:
                    grade = "🟠 C"
                else:
                    grade = "🔴 D"
                
                results.append({
                    'Coin': symbol.split('/')[0],
                    'SmartScore': smart_score,
                    'Grade': grade,
                    'TrendScore': trend_score,
                    'VolumeScore': volume_score,
                    'VolatilityScore': volatility_score,
                    'RSI': rsi,
                    'Price': price
                })
                
            except:
                results.append({
                    'Coin': symbol.split('/')[0],
                    'SmartScore': 0,
                    'Grade': '❓',
                    'TrendScore': 0,
                    'VolumeScore': 0,
                    'VolatilityScore': 0,
                    'RSI': 0,
                    'Price': 0
                })
        
        # Skora göre sırala
        results = sorted(results, key=lambda x: x['SmartScore'], reverse=True)
        return results
        
    except Exception as e:
        return []


# ==================== PİYASA DERİNLİĞİ VE DUYGU MODÜLLERİ ====================

@st.cache_data(ttl=3600, show_spinner=False)  # Makro veriler: 1 saat
def fetch_liquidity_proxy():
    """
    Piyasa Sentiment Göstergesi - Fiyat momentumu bazlı.
    (Binance Futures Türkiye'den erişilemediği için alternatif yöntem)
    
    Returns:
        list: Her coin için sentiment verisi
    """
    
    TOP_COINS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT',
                 'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'DOT/USDT', 'POL/USDT']
    
    try:
        exchange = ccxt.kucoin({'enableRateLimit': True})
        results = []
        
        for symbol in TOP_COINS:
            try:
                # Son 24 saat ve 1 saatlik veriler
                ticker = exchange.fetch_ticker(symbol)
                change_24h = ticker.get('percentage', 0) or 0
                
                # Son 4 saatlik mumları çek
                ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=4)
                if len(ohlcv) >= 4:
                    recent_closes = [c[4] for c in ohlcv]
                    momentum = ((recent_closes[-1] - recent_closes[0]) / recent_closes[0]) * 100
                else:
                    momentum = change_24h / 6  # Tahmini
                
                # Simüle edilmiş "Funding Rate" (momentum bazlı)
                simulated_rate = momentum * 0.01  # Ölçeklendirme
                
                # Sentiment belirleme
                if change_24h > 5 and momentum > 1:
                    sentiment = "🔴 Aşırı Long"
                    risk = "Düşüş Riski"
                elif change_24h < -5 and momentum < -1:
                    sentiment = "🟢 Aşırı Short"
                    risk = "Squeeze Fırsatı"
                elif change_24h > 2:
                    sentiment = "🟠 Long Baskın"
                    risk = "Dikkat"
                elif change_24h < -2:
                    sentiment = "🟢 Short Baskın"
                    risk = "Fırsat Olabilir"
                else:
                    sentiment = "🟡 Nötr"
                    risk = "Dengeli"
                
                results.append({
                    'Coin': symbol.split('/')[0],
                    'FundingRate': simulated_rate,
                    'Sentiment': sentiment,
                    'Risk': risk
                })
                
            except:
                results.append({
                    'Coin': symbol.split('/')[0],
                    'FundingRate': 0,
                    'Sentiment': '❓ Veri Yok',
                    'Risk': '-'
                })
        
        return results
        
    except Exception as e:
        return []


@st.cache_data(ttl=300, show_spinner=False)
def calculate_orderbook_imbalance():
    """
    Order Book Imbalance - Alış/Satış duvar analizi.
    Bid/Ask Ratio: ((Bids - Asks) / (Bids + Asks)) * 100
    
    Returns:
        list: Her coin için imbalance verisi
    """
    
    TOP_COINS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT',
                 'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'DOT/USDT', 'POL/USDT']
    
    try:
        exchange = ccxt.kucoin({'enableRateLimit': True})
        results = []
        
        for symbol in TOP_COINS:
            try:
                # Order book çek (ilk 20 kademe)
                orderbook = exchange.fetch_order_book(symbol, limit=20)
                
                # Toplam bids ve asks hacmi
                total_bids = sum([bid[1] for bid in orderbook['bids']])
                total_asks = sum([ask[1] for ask in orderbook['asks']])
                
                # Imbalance hesapla
                if (total_bids + total_asks) > 0:
                    imbalance = ((total_bids - total_asks) / (total_bids + total_asks)) * 100
                else:
                    imbalance = 0
                
                # Durum belirleme
                if imbalance > 10:
                    status = "🟢 Alıcılar Güçlü"
                elif imbalance < -10:
                    status = "🔴 Satıcılar Baskın"
                else:
                    status = "🟡 Dengeli"
                
                results.append({
                    'Coin': symbol.split('/')[0],
                    'Imbalance': imbalance,
                    'TotalBids': total_bids,
                    'TotalAsks': total_asks,
                    'Status': status
                })
                
            except:
                results.append({
                    'Coin': symbol.split('/')[0],
                    'Imbalance': 0,
                    'TotalBids': 0,
                    'TotalAsks': 0,
                    'Status': '❓ Veri Yok'
                })
        
        return results
        
    except Exception as e:
        return []


@st.cache_data(ttl=300, show_spinner=False)
def detect_volume_anomalies():
    """
    Anomali Radarı - Hacim patlamalarını tespit et.
    3-Sigma kuralı: Son hacim > Ortalama * 3 ise anomali.
    
    Returns:
        list: Her coin için anomali verisi
    """
    
    TOP_COINS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT',
                 'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'DOT/USDT', 'POL/USDT']
    
    try:
        exchange = ccxt.kucoin({'enableRateLimit': True})
        results = []
        
        for symbol in TOP_COINS:
            try:
                # Son 24 saatlik 1h mumları çek
                ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=24)
                volumes = np.array([c[5] for c in ohlcv])
                
                # Son 1 saatlik hacim
                last_volume = volumes[-1] if len(volumes) > 0 else 0
                
                # Ortalama ve standart sapma
                avg_volume = np.mean(volumes)
                std_volume = np.std(volumes)
                
                # Z-Score hesapla
                z_score = (last_volume - avg_volume) / std_volume if std_volume > 0 else 0
                
                # Anomali tespiti (3-sigma)
                if z_score >= 3:
                    anomaly = "🚨 PATLAMA!"
                    is_anomaly = True
                elif z_score >= 2:
                    anomaly = "⚠️ Yüksek"
                    is_anomaly = False
                else:
                    anomaly = "✅ Normal"
                    is_anomaly = False
                
                # Oran hesapla
                ratio = last_volume / avg_volume if avg_volume > 0 else 1
                
                results.append({
                    'Coin': symbol.split('/')[0],
                    'LastVolume': last_volume,
                    'AvgVolume': avg_volume,
                    'Ratio': ratio,
                    'ZScore': z_score,
                    'Anomaly': anomaly,
                    'IsAnomaly': is_anomaly
                })
                
            except:
                results.append({
                    'Coin': symbol.split('/')[0],
                    'LastVolume': 0,
                    'AvgVolume': 0,
                    'Ratio': 0,
                    'ZScore': 0,
                    'Anomaly': '❓ Veri Yok',
                    'IsAnomaly': False
                })
        
        return results
        
    except Exception as e:
        return []


# ==================== KESKİN NİŞANCI MODÜLÜ (SNIPER MODE) ====================

@st.cache_data(ttl=600, show_spinner=False)
def calculate_channel_bender():
    """
    Channel Bender - Fiyatın kanal sınırlarından sapma skoru.
    Bollinger Bantları üzerinden hesaplanır.
    
    Skor > 1.0: Aşırı alım (kanal üstü taşma)
    Skor < -1.0: Aşırı satım (kanal altı taşma)
    
    Returns:
        list: Her coin için sapma skoru
    """
    
    TOP_COINS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT',
                 'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'DOT/USDT', 'POL/USDT']
    
    try:
        exchange = ccxt.kucoin({'enableRateLimit': True})
        results = []
        
        for symbol in TOP_COINS:
            try:
                # Son 20 periyot (4h mumları)
                ohlcv = exchange.fetch_ohlcv(symbol, '4h', limit=20)
                closes = np.array([c[4] for c in ohlcv])
                
                # Bollinger Bantları
                middle = np.mean(closes)  # SMA(20)
                std = np.std(closes)
                upper = middle + (2 * std)
                lower = middle - (2 * std)
                
                # Mevcut fiyat
                current_price = closes[-1]
                
                # Sapma Skoru: (Fiyat - Orta) / (Üst - Orta)
                if (upper - middle) > 0:
                    deviation_score = (current_price - middle) / (upper - middle)
                else:
                    deviation_score = 0
                
                # Yorum belirleme
                if deviation_score > 1.0:
                    status = "🔴 Aşırı Alım"
                    zone = "Kanal Üstü"
                elif deviation_score > 0.5:
                    status = "🟠 Yüksek"
                    zone = "Üst Bölge"
                elif deviation_score < -1.0:
                    status = "🟢 Aşırı Satım"
                    zone = "Kanal Altı"
                elif deviation_score < -0.5:
                    status = "🟢 Düşük"
                    zone = "Alt Bölge"
                else:
                    status = "🟡 Dengeli"
                    zone = "Orta Bölge"
                
                results.append({
                    'Coin': symbol.split('/')[0],
                    'Price': current_price,
                    'Middle': middle,
                    'Upper': upper,
                    'Lower': lower,
                    'DeviationScore': deviation_score,
                    'Status': status,
                    'Zone': zone
                })
                
            except:
                results.append({
                    'Coin': symbol.split('/')[0],
                    'Price': 0,
                    'Middle': 0,
                    'Upper': 0,
                    'Lower': 0,
                    'DeviationScore': 0,
                    'Status': '❓ Veri Yok',
                    'Zone': '-'
                })
        
        return results
        
    except Exception as e:
        return []


@st.cache_data(ttl=300, show_spinner=False)
def detect_pump_corrections():
    """
    Pump & Correction Radar - Ani yükselen coinlere Fibonacci düzeltme seviyeleri.
    Son 1 saatte %5+ yükselenler için Fib seviyeleri hesaplar.
    
    Returns:
        list: Pumped coinler ve Fibonacci seviyeleri
    """
    
    TOP_COINS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT',
                 'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'DOT/USDT', 'POL/USDT']
    
    PUMP_THRESHOLD = 5.0  # %5 eşik
    
    try:
        exchange = ccxt.kucoin({'enableRateLimit': True})
        results = []
        
        for symbol in TOP_COINS:
            try:
                # Son 24 saatlik veriler
                ohlcv_24h = exchange.fetch_ohlcv(symbol, '1h', limit=24)
                
                # Son 1 saatlik değişim
                if len(ohlcv_24h) >= 2:
                    close_now = ohlcv_24h[-1][4]
                    close_1h_ago = ohlcv_24h[-2][4]
                    change_1h = ((close_now - close_1h_ago) / close_1h_ago) * 100
                else:
                    change_1h = 0
                
                # Pump kontrolü
                if change_1h >= PUMP_THRESHOLD:
                    # 24h Min/Max
                    highs = [c[2] for c in ohlcv_24h]
                    lows = [c[3] for c in ohlcv_24h]
                    high_24h = max(highs)
                    low_24h = min(lows)
                    
                    range_24h = high_24h - low_24h
                    
                    # Fibonacci Seviyeleri
                    fib_382 = high_24h - (range_24h * 0.382)
                    fib_500 = high_24h - (range_24h * 0.500)
                    fib_618 = high_24h - (range_24h * 0.618)
                    
                    results.append({
                        'Coin': symbol.split('/')[0],
                        'Price': close_now,
                        'Change1H': change_1h,
                        'High24H': high_24h,
                        'Low24H': low_24h,
                        'Fib382': fib_382,
                        'Fib500': fib_500,
                        'Fib618': fib_618,
                        'IsPumping': True
                    })
                    
            except:
                continue
        
        # Değişime göre sırala
        results = sorted(results, key=lambda x: x['Change1H'], reverse=True)
        return results
        
    except Exception as e:
        return []


@st.cache_data(ttl=600, show_spinner=False)
def calculate_support_resistance():
    """
    Otomatik Destek/Direnç - Local Min/Max noktalarından hesaplama.
    Son 50 mumda en yakın destek ve direnç seviyeleri.
    
    Returns:
        list: Her coin için destek ve direnç seviyeleri
    """
    from scipy.signal import argrelextrema
    
    TOP_COINS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT',
                 'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'DOT/USDT', 'POL/USDT']
    
    try:
        exchange = ccxt.kucoin({'enableRateLimit': True})
        results = []
        
        for symbol in TOP_COINS:
            try:
                # Son 50 mum (4h)
                ohlcv = exchange.fetch_ohlcv(symbol, '4h', limit=50)
                highs = np.array([c[2] for c in ohlcv])
                lows = np.array([c[3] for c in ohlcv])
                closes = np.array([c[4] for c in ohlcv])
                
                current_price = closes[-1]
                
                # Local maxima (direnç seviyeleri)
                local_max_idx = argrelextrema(highs, np.greater, order=3)[0]
                resistance_levels = highs[local_max_idx] if len(local_max_idx) > 0 else []
                
                # Local minima (destek seviyeleri)
                local_min_idx = argrelextrema(lows, np.less, order=3)[0]
                support_levels = lows[local_min_idx] if len(local_min_idx) > 0 else []
                
                # En yakın direnç (fiyatın üstündekiler)
                resistances_above = [r for r in resistance_levels if r > current_price]
                nearest_resistance = min(resistances_above) if resistances_above else highs.max()
                
                # En yakın destek (fiyatın altındakiler)
                supports_below = [s for s in support_levels if s < current_price]
                nearest_support = max(supports_below) if supports_below else lows.min()
                
                # Fiyatın konumu
                range_sr = nearest_resistance - nearest_support
                if range_sr > 0:
                    position_pct = ((current_price - nearest_support) / range_sr) * 100
                else:
                    position_pct = 50
                
                results.append({
                    'Coin': symbol.split('/')[0],
                    'Price': current_price,
                    'Support': nearest_support,
                    'Resistance': nearest_resistance,
                    'PositionPct': position_pct,
                    'RangePct': (range_sr / current_price) * 100 if current_price > 0 else 0
                })
                
            except:
                results.append({
                    'Coin': symbol.split('/')[0],
                    'Price': 0,
                    'Support': 0,
                    'Resistance': 0,
                    'PositionPct': 50,
                    'RangePct': 0
                })
        
        return results
        
    except Exception as e:
        return []


@st.cache_data(ttl=900, show_spinner=False)
def fetch_stock_data(symbol: str, period: str = "6mo"):
    """Yahoo Finance'den hisse verisi."""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
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


@st.cache_data(ttl=3600, show_spinner=False)  # Makro veriler: 1 saat
def fetch_macro_data():
    """Genişletilmiş makro ekonomi verileri."""
    
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


@st.cache_data(ttl=3600, show_spinner=False)  # Makro veriler: 1 saat
def fetch_yield_curve_data():
    """Getiri eğrisi verisi (10Y-2Y spread)."""
    
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
        df = pd.DataFrame(returns_data)
        
        # Korelasyon matrisi
        corr_matrix = df.corr()
        
        return corr_matrix, None
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=600, show_spinner=False)
def fetch_correlation_data(crypto_symbol: str = "BTC-USD", days: int = 90):
    """DXY ve Kripto arasındaki korelasyonu hesaplar."""
    
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
    
    # ==================== AKSİYON MERKEZİ ====================
    with st.container():
        # AI ve Makro verilerini kontrol et
        ai_prob = None
        risk_score = st.session_state.get('risk_score', 50)
        market_regime = st.session_state.get('market_regime', 'KARIŞIK')
        
        if 'xgb_model' in st.session_state and st.session_state.xgb_model is not None:
            try:
                last_row = st.session_state.xgb_last_row
                proba = st.session_state.xgb_model.predict_proba(last_row)[0]
                ai_prob = proba[1] * 100
            except:
                ai_prob = None
        
        # Karar mantığı
        if ai_prob is not None and ai_prob > 55 and risk_score > 60:
            # YEŞİL: Olumlu koşullar
            st.markdown("""
            <div style="background: linear-gradient(135deg, #00C85322, #00C85344); border: 3px solid #00C853; border-radius: 15px; padding: 25px; margin-bottom: 20px;">
                <h2 style="color: #00C853; margin: 0; text-align: center;">✅ YATIRIM İÇİN UYGUN KOŞULLAR</h2>
                <p style="color: #888; text-align: center; margin: 10px 0;">AI tahmini olumlu, makro riskler düşük. Pozisyon açmak için uygun ortam.</p>
            </div>
            """, unsafe_allow_html=True)
        elif ai_prob is not None and ai_prob < 45 or risk_score < 40:
            # KIRMIZI: Riskli koşullar
            st.markdown("""
            <div style="background: linear-gradient(135deg, #FF174422, #FF174444); border: 3px solid #FF1744; border-radius: 15px; padding: 25px; margin-bottom: 20px;">
                <h2 style="color: #FF1744; margin: 0; text-align: center;">⚠️ RİSK YÜKSEK - KORUNMA MODU</h2>
                <p style="color: #888; text-align: center; margin: 10px 0;">AI tahmini olumsuz veya makro riskler yüksek. Nakit/altın pozisyonu düşünün.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # SARI: Nötr/Karışık
            st.markdown("""
            <div style="background: linear-gradient(135deg, #FF980022, #FF980044); border: 3px solid #FF9800; border-radius: 15px; padding: 25px; margin-bottom: 20px;">
                <h2 style="color: #FF9800; margin: 0; text-align: center;">🔄 KARIŞIK SİNYALLER - DİKKATLİ OLUN</h2>
                <p style="color: #888; text-align: center; margin: 10px 0;">Piyasa yön arıyor. Küçük pozisyonlar, stop-loss kullanın.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Hızlı göstergeler
        quick_cols = st.columns(4)
        
        with quick_cols[0]:
            if ai_prob is not None:
                ai_color = "#00C853" if ai_prob > 55 else "#FF1744" if ai_prob < 45 else "#FF9800"
                st.markdown(f"""
                <div style="text-align: center; padding: 15px; background: {ai_color}22; border-radius: 10px;">
                    <p style="margin: 0; color: #888; font-size: 0.8rem;">🤖 AI Tahmini</p>
                    <h2 style="color: {ai_color}; margin: 5px 0;">{ai_prob:.0f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("🤖 AI: Model eğitilmedi")
        
        with quick_cols[1]:
            risk_color = "#00C853" if risk_score > 60 else "#FF1744" if risk_score < 40 else "#FF9800"
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; background: {risk_color}22; border-radius: 10px;">
                <p style="margin: 0; color: #888; font-size: 0.8rem;">🧭 Risk Skoru</p>
                <h2 style="color: {risk_color}; margin: 5px 0;">{risk_score:.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with quick_cols[2]:
            regime_color = "#00C853" if "GOLD" in market_regime else "#FF1744" if "RESES" in market_regime else "#FF9800"
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; background: {regime_color}22; border-radius: 10px;">
                <p style="margin: 0; color: #888; font-size: 0.8rem;">🌍 Piyasa Rejimi</p>
                <p style="color: {regime_color}; margin: 5px 0; font-size: 0.9rem; font-weight: bold;">{market_regime[:15]}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with quick_cols[3]:
            # SHAP en önemli 3 faktör
            if 'xgb_importance' in st.session_state:
                top3 = st.session_state.xgb_importance.head(3)['feature'].tolist()
                factors_text = "<br>".join([f"• {f[:15]}" for f in top3])
                st.markdown(f"""
                <div style="text-align: center; padding: 15px; background: #2196F322; border-radius: 10px;">
                    <p style="margin: 0; color: #888; font-size: 0.8rem;">📊 En Önemli Faktörler</p>
                    <p style="color: #2196F3; margin: 5px 0; font-size: 0.7rem;">{factors_text}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="text-align: center; padding: 15px; background: #9E9E9E22; border-radius: 10px;">
                    <p style="margin: 0; color: #888; font-size: 0.8rem;">📊 En Önemli Faktörler</p>
                    <p style="color: #9E9E9E; margin: 5px 0; font-size: 0.8rem;">Model eğitilmedi</p>
                </div>
                """, unsafe_allow_html=True)
    
    st.divider()
    
    # ==================== 📡 PİYASA RADARI ====================
    st.markdown("### 📡 Piyasa Radarı")
    st.caption("Top 10 Majör Coin - TrendString (4H) ve Nakit Akış Analizi")
    
    with st.spinner("Piyasa radarı verileri yükleniyor..."):
        radar_data = fetch_market_radar_data()
    
    if radar_data:
        # DataFrame oluştur
        df_radar = pd.DataFrame(radar_data)
        
        # Görüntülenecek sütunları seç ve formatla
        df_display = df_radar[['Coin', 'Fiyat', 'TrendString', 'InOut', '24s Değişim']].copy()
        
        # Fiyat formatlama
        df_display['Fiyat'] = df_display['Fiyat'].apply(
            lambda x: f"${x:,.0f}" if x > 100 else f"${x:,.4f}" if x < 1 else f"${x:,.2f}"
        )
        
        # 24s Değişim formatlama
        df_display['24s Değişim'] = df_display['24s Değişim'].apply(lambda x: f"{x:+.2f}%")
        
        # TrendString renkli görüntüleme
        def color_trend(val):
            colored = ""
            for char in val:
                if char == '+':
                    colored += '<span style="color:#00C853;font-weight:bold;">+</span>'
                elif char == '-':
                    colored += '<span style="color:#FF1744;font-weight:bold;">-</span>'
                else:
                    colored += char
            return colored
        
        # InOut renkli görüntüleme
        def color_inout(val):
            if 'Giriş' in val:
                return f'<span style="color:#00C853;">{val}</span>'
            elif 'Çıkış' in val:
                return f'<span style="color:#FF1744;">{val}</span>'
            return val
        
        # Pandas Styler ile formatlama
        def highlight_trend(val):
            color = "#00C853" if '+' in val else "#FF1744"
            return f'color: {color}; font-family: monospace; font-weight: bold;'

        def highlight_change(val):
            try:
                # % işaretini kaldırıp sayıya çevir
                num = float(val.replace('%', '').replace('+', ''))
                color = "#00C853" if num >= 0 else "#FF1744"
                return f'color: {color}'
            except:
                return ''

        # Display için yeni DF hazırla (Ham verilerden)
        df_radar_view = df_radar[['Coin', 'Fiyat', 'TrendString', 'InOut', '24s Değişim']].copy()
        
        # Kolon isimlerini Türkçeleştir
        df_radar_view.columns = ['Coin', 'Fiyat ($)', 'Trend (4H)', 'Nakit Akış', '24H (%)']

        # Styler uygula (CSS yerine)
        st.dataframe(
            df_radar_view,
            column_config={
                "Coin": st.column_config.TextColumn("Coin", width="small"),
                "Fiyat ($)": st.column_config.NumberColumn("Fiyat", format="$%.2f"),
                "Trend (4H)": st.column_config.TextColumn("Trend", width="medium"), # TrendString özel font gerektirir ama dataframe kısıtlı
                "Nakit Akış": st.column_config.TextColumn("Nakit Akış", width="medium"),
                "24H (%)": st.column_config.NumberColumn("24H", format="%.2f%%")
            },
            use_container_width=True,
            hide_index=True
        )
        
        with st.expander("💡 Piyasa Radarı Nasıl Okunur?"):
            st.markdown("""
            **TrendString (Trend Dizisi)**: Son 5 adet 4 saatlik mumun yönü.
            - `+` = Yeşil mum (kapanış > açılış)
            - `-` = Kırmızı mum (kapanış < açılış)
            - Örnek: `++--+` = 3 yükseliş, 2 düşüş
            
            **Nakit Akış (InOut)**: Hacim ağırlıklı fiyat değişimi.
            - 🟢 **Güçlü Giriş**: Yüksek hacimle yükseliş (para giriyor)
            - 🔴 **Güçlü Çıkış**: Yüksek hacimle düşüş (para çıkıyor)
            - ⚪ **Nötr**: Dengeli durum
            """)
    else:
        st.warning("Piyasa radarı verisi yüklenemedi.")
    
    st.divider()
    
    # ==================== 🔍 DERİN ANALİZ LABORATUVARI ====================
    with st.expander("🔍 Derin Analiz Laboratuvarı", expanded=False):
        st.caption("Gelişmiş teknik analiz araçları: Korelasyon, Smart Score, Sıkışma Analizi")
        
        lab_tabs = st.tabs(["📊 Smart Score", "🔥 Sıkışma Analizi", "🌡️ Korelasyon Haritası"])
        
        # ===== SMART SCORE TAB =====
        with lab_tabs[0]:
            st.markdown("#### 📊 Smart Score Sıralaması")
            st.caption("Trend (40%) + Hacim (40%) + Volatilite (20%) = Toplam Kalite Puanı")
            
            with st.spinner("Smart Score hesaplanıyor..."):
                smart_data = calculate_smart_scores()
            
            if smart_data:
                df_ss = pd.DataFrame(smart_data)
                # İstenilen sütunları seç
                df_ss = df_ss[['Coin', 'SmartScore', 'Grade', 'TrendScore', 'VolumeScore', 'RSI']]
                
                st.dataframe(
                    df_ss,
                    column_config={
                        "Coin": st.column_config.TextColumn("Coin", width="small"),
                        "SmartScore": st.column_config.NumberColumn("Smart Score", format="%.0f"),
                        "Grade": st.column_config.TextColumn("Grade", width="small"),
                        "TrendScore": st.column_config.NumberColumn("Trend", format="%.0f"),
                        "VolumeScore": st.column_config.NumberColumn("Hacim", format="%.0f"),
                        "RSI": st.column_config.NumberColumn("RSI", format="%.0f")
                    },
                    use_container_width=True,
                    hide_index=True
                )
                
                st.markdown("""
                **Grade Sistemi**: 🟢 A (≥75) | 🟡 B (≥60) | 🟠 C (≥40) | 🔴 D (<40)
                """)
            else:
                st.warning("Smart Score verisi yüklenemedi.")
        
        # ===== SIKIŞMA ANALİZİ TAB =====
        with lab_tabs[1]:
            st.markdown("#### 🔥 Volatilite Sıkışması (Bollinger Bandwidth)")
            st.caption("Düşük bandwidth = Fiyat patlayabilir!")
            
            with st.spinner("Sıkışma analizi yapılıyor..."):
                squeeze_data = calculate_squeeze_volatility()
            
            if squeeze_data:
                # Sıkışan coinleri öne çıkar
                alerts = [s for s in squeeze_data if s['SqueezeAlert']]
                
                if alerts:
                    st.warning(f"⚠️ {len(alerts)} coin sıkışma bölgesinde!")
                
                df_sq = pd.DataFrame(squeeze_data)
                df_sq = df_sq.sort_values(by='Bandwidth')
                df_sq = df_sq[['Coin', 'Bandwidth', 'SqueezeStatus']]
                
                st.dataframe(
                    df_sq,
                    column_config={
                        "Coin": st.column_config.TextColumn("Coin", width="small"),
                        "Bandwidth": st.column_config.NumberColumn("Bandwidth %", format="%.2f%%"),
                        "SqueezeStatus": st.column_config.TextColumn("Durum", width="medium")
                    },
                    use_container_width=True,
                    hide_index=True
                )
                
                st.markdown("""
                **Yorum**: Bandwidth %4'ün altına düştüğünde fiyat genellikle güçlü bir hareket yapar (yukarı veya aşağı).
                """)
            else:
                st.warning("Sıkışma verisi yüklenemedi.")
        
        # ===== KORELASYON HARİTASI TAB =====
        with lab_tabs[2]:
            st.markdown("#### 🌡️ 30 Günlük Korelasyon Isı Haritası")
            st.caption("Coinler arasındaki fiyat ilişkisi (-1 ile +1 arası)")
            
            with st.spinner("Korelasyon matrisi hesaplanıyor..."):
                corr_matrix, coins = fetch_correlation_matrix()
            
            if corr_matrix is not None and len(coins) > 0:
                
                fig = px.imshow(
                    corr_matrix,
                    labels=dict(x="Coin", y="Coin", color="Korelasyon"),
                    x=coins,
                    y=coins,
                    color_continuous_scale='RdBu_r',  # Kırmızı-Beyaz-Mavi
                    zmin=-1,
                    zmax=1,
                    aspect='auto'
                )
                
                fig.update_layout(
                    template='plotly_dark',
                    height=400,
                    margin=dict(l=0, r=0, t=30, b=0),
                    title=None
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                | Değer | Anlam |
                |-------|-------|
                | **+1.0** | Mükemmel pozitif korelasyon (beraber hareket) |
                | **0.0** | Korelasyon yok (bağımsız) |
                | **-1.0** | Negatif korelasyon (ters hareket) |
                """)
            else:
                st.warning("Korelasyon verisi yüklenemedi.")
    
    st.divider()
    
    # ==================== 📡 PİYASA DERİNLİĞİ VE DUYGU ====================
    with st.expander("📡 Piyasa Derinliği ve Duygu", expanded=False):
        st.caption("Futures sentiment, emir defteri dengesizliği ve hacim anomalileri")
        
        depth_tabs = st.tabs(["💰 Funding Rate", "📊 Order Book", "🚨 Anomali Radarı"])
        
        # ===== FUNDING RATE TAB =====
        with depth_tabs[0]:
            st.markdown("#### 💰 Funding Rate Analizi (Futures Sentiment)")
            st.caption("Long/Short pozisyon yığılmasını gösterir")
            
            with st.spinner("Funding rate verileri çekiliyor..."):
                funding_data = fetch_funding_rates()
            
            if funding_data:
                df_fr = pd.DataFrame(funding_data)
                df_fr = df_fr[['Coin', 'FundingRate', 'Sentiment', 'Risk']]
                
                st.dataframe(
                    df_fr,
                    column_config={
                        "Coin": st.column_config.TextColumn("Coin", width="small"),
                        "FundingRate": st.column_config.NumberColumn("Funding Rate", format="%.4f%%"),
                        "Sentiment": st.column_config.TextColumn("Sentiment", width="medium"),
                        "Risk": st.column_config.TextColumn("Risk", width="medium")
                    },
                    use_container_width=True,
                    hide_index=True
                )
                
                st.markdown("""
                **Yorumlama**:
                - 🔴 **Aşırı Long** (>0.01%): Çok fazla yükseliş beklentisi, düşüş riski
                - 🟢 **Aşırı Short** (<0%): Short squeeze fırsatı olabilir
                - 🟡 **Nötr**: Dengeli piyasa
                """)
            else:
                st.warning("Funding rate verisi yüklenemedi.")
        
        # ===== ORDER BOOK TAB =====
        with depth_tabs[1]:
            st.markdown("#### 📊 Emir Defteri Dengesizliği")
            st.caption("Alış/Satış duvarları (ilk 20 kademe)")
            
            with st.spinner("Order book verileri çekiliyor..."):
                orderbook_data = calculate_orderbook_imbalance()
            
            if orderbook_data:
                df_ob = pd.DataFrame(orderbook_data)
                df_ob = df_ob.sort_values(by='Imbalance', key=abs, ascending=False)
                df_ob = df_ob[['Coin', 'Imbalance', 'Status']]
                
                st.dataframe(
                    df_ob,
                    column_config={
                        "Coin": st.column_config.TextColumn("Coin", width="small"),
                        "Imbalance": st.column_config.NumberColumn("Imbalance", format="%+.1f%%"),
                        "Status": st.column_config.TextColumn("Durum", width="medium")
                    },
                    use_container_width=True,
                    hide_index=True
                )
                
                st.markdown("""
                **Formül**: `((Bids - Asks) / (Bids + Asks)) × 100`
                - **Pozitif (+)**: Alıcılar güçlü, yukarı baskı
                - **Negatif (-)**: Satıcılar baskın, aşağı baskı
                """)
            else:
                st.warning("Order book verisi yüklenemedi.")
        
        # ===== ANOMALİ RADARI TAB =====
        with depth_tabs[2]:
            st.markdown("#### 🚨 Hacim Anomali Radarı")
            st.caption("3-Sigma kuralı ile pump/dump tespiti")
            
            with st.spinner("Hacim verileri analiz ediliyor..."):
                anomaly_data = detect_volume_anomalies()
            
            if anomaly_data:
                # Anomali uyarıları
                anomalies = [a for a in anomaly_data if a['IsAnomaly']]
                if anomalies:
                    for a in anomalies:
                        st.error(f"🚨 **{a['Coin']}**: Hacim patlaması tespit edildi! (Oran: {a['Ratio']:.1f}x)")
                
                df_an = pd.DataFrame(anomaly_data)
                df_an = df_an.sort_values(by='ZScore', ascending=False)
                df_an = df_an[['Coin', 'Ratio', 'ZScore', 'Anomaly']]
                
                st.dataframe(
                    df_an,
                    column_config={
                        "Coin": st.column_config.TextColumn("Coin", width="small"),
                        "Ratio": st.column_config.NumberColumn("Hacim Oranı", format="%.2fx"),
                        "ZScore": st.column_config.NumberColumn("Z-Score", format="%.1fσ"),
                        "Anomaly": st.column_config.TextColumn("Durum", width="medium")
                    },
                    use_container_width=True,
                    hide_index=True
                )
                
                st.markdown("""
                **Z-Score Yorumu**:
                - **≥3σ**: 🚨 Anormal hacim patlaması (Pump/Dump olabilir)
                - **≥2σ**: ⚠️ Ortalama üstü hacim
                - **<2σ**: ✅ Normal hacim
                """)
            else:
                st.warning("Hacim verisi yüklenemedi.")
    
    st.divider()
    
    # ==================== 🎯 KESKİN NİŞANCI MODÜLÜ ====================
    with st.expander("🎯 Keskin Nişancı Modülü (Sniper Mode)", expanded=False):
        st.caption("Kanal sapmaları, pump tespiti ve destek/direnç seviyeleri")
        
        sniper_tabs = st.tabs(["📐 Kanal Bükücü", "🚀 Pump Radarı", "⚡ Destek/Direnç"])
        
        # ===== KANAL BÜKÜCÜ TAB =====
        with sniper_tabs[0]:
            st.markdown("#### 📐 Kanal Bükücü (Channel Bender)")
            st.caption("Fiyatın Bollinger kanalından sapma skoru")
            
            with st.spinner("Kanal analizi yapılıyor..."):
                channel_data = calculate_channel_bender()
            
            if channel_data:
                # Aşırı durumları öne çıkar
                extremes = [c for c in channel_data if abs(c['DeviationScore']) > 1.0]
                if extremes:
                    for e in extremes:
                        color = "red" if e['DeviationScore'] > 0 else "green"
                        st.markdown(f":{color}[**{e['Coin']}**: {e['Status']} (Skor: {e['DeviationScore']:.2f})]")
                
                df_ch = pd.DataFrame(channel_data)
                df_ch = df_ch.sort_values(by='DeviationScore', key=abs, ascending=False)
                df_ch = df_ch[['Coin', 'Price', 'DeviationScore', 'Status', 'Zone']]
                
                st.dataframe(
                    df_ch,
                    column_config={
                        "Coin": st.column_config.TextColumn("Coin", width="small"),
                        "Price": st.column_config.NumberColumn("Fiyat", format="$%.2f"),
                        "DeviationScore": st.column_config.NumberColumn("Sapma Skoru", format="%+.2f"),
                        "Status": st.column_config.TextColumn("Durum", width="medium"),
                        "Zone": st.column_config.TextColumn("Bölge", width="small")
                    },
                    use_container_width=True,
                    hide_index=True
                )
                
                st.markdown("""
                **Yorumlama**:
                - **> +1.0**: 🔴 Kanal üstüne taşmış (aşırı alım, dönüş beklentisi)
                - **< -1.0**: 🟢 Kanal altına düşmüş (aşırı satım, tepki beklentisi)
                - **-0.5 ile +0.5**: 🟡 Dengeli bölge
                """)
            else:
                st.warning("Kanal verisi yüklenemedi.")
        
        # ===== PUMP RADARI TAB =====
        with sniper_tabs[1]:
            st.markdown("#### 🚀 Pump & Düzeltme Radarı")
            st.caption("Son 1 saatte %5+ yükselen coinler ve Fibonacci seviyeleri")
            
            with st.spinner("Pump taraması yapılıyor..."):
                pump_data = detect_pump_corrections()
            
            if pump_data:
                st.success(f"🚨 **{len(pump_data)} coin pump yapıyor!**")
                
                for coin in pump_data:
                    st.markdown(f"""
                    <div style="background: #2a2a2a; border-left: 3px solid #FF9800; padding: 15px; margin: 10px 0; border-radius: 5px;">
                        <h4 style="margin: 0; color: #FF9800;">🚀 {coin['Coin']} (+{coin['Change1H']:.1f}%)</h4>
                        <p style="margin: 5px 0; color: #fff;">Fiyat: <strong>${coin['Price']:,.2f}</strong></p>
                        <p style="margin: 5px 0; color: #888;">24H Range: ${coin['Low24H']:,.2f} - ${coin['High24H']:,.2f}</p>
                        <hr style="border-color: #444;">
                        <p style="margin: 5px 0; color: #00C853;">📍 Fib 0.382 (Destek 1): <strong>${coin['Fib382']:,.2f}</strong></p>
                        <p style="margin: 5px 0; color: #FFD700;">📍 Fib 0.500 (Orta): <strong>${coin['Fib500']:,.2f}</strong></p>
                        <p style="margin: 5px 0; color: #00C853;">📍 Fib 0.618 (Altın Oran): <strong>${coin['Fib618']:,.2f}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("""
                **Strateji**: Pump sonrası düzeltmede Fib 0.618 seviyesi güçlü destek olabilir.
                """)
            else:
                st.info("🔍 Son 1 saatte %5+ yükseliş gösteren coin yok.")
        
        # ===== DESTEK/DİRENÇ TAB =====
        with sniper_tabs[2]:
            st.markdown("#### ⚡ Otomatik Destek & Direnç")
            st.caption("Local Min/Max noktalarından hesaplanmış seviyeler")
            
            with st.spinner("Seviyeler hesaplanıyor..."):
                sr_data = calculate_support_resistance()
            
            if sr_data:
                df_sr = pd.DataFrame(sr_data)
                
                # Konum yazısı oluştur
                def get_position_text(pct):
                    if pct > 70: return f"Dirence Yakın ({pct:.0f}%)"
                    elif pct < 30: return f"Desteğe Yakın ({pct:.0f}%)"
                    return f"Ortada ({pct:.0f}%)"
                
                df_sr['Konum'] = df_sr['PositionPct'].apply(get_position_text)
                df_sr = df_sr[['Coin', 'Support', 'Price', 'Resistance', 'Konum']]
                
                st.dataframe(
                    df_sr,
                    column_config={
                        "Coin": st.column_config.TextColumn("Coin", width="small"),
                        "Support": st.column_config.NumberColumn("Destek", format="$%.2f"),
                        "Price": st.column_config.NumberColumn("Fiyat", format="$%.2f"),
                        "Resistance": st.column_config.NumberColumn("Direnç", format="$%.2f"),
                        "Konum": st.column_config.TextColumn("Konum (%)", width="medium")
                    },
                    use_container_width=True,
                    hide_index=True
                )
                
                st.markdown("""
                **Okuma**: Fiyat desteğe yakınsa alım fırsatı, dirence yakınsa satış baskısı beklenebilir.
                """)
            else:
                st.warning("Destek/Direnç verisi yüklenemedi.")
    
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
    
    # İnsan dostu makro özet
    with st.expander("💡 Bu Veriler Ne Anlama Geliyor?"):
        st.markdown("""
        | Gösterge | Basit Adı | Yukarı ⬆️ | Aşağı ⬇️ |
        |----------|-----------|-----------|----------|
        | **DXY** | 💵 Doların Gücü | Kripto için kötü | Kripto için iyi |
        | **VIX** | 😱 Korku Endeksi | Piyasa panik modda | Piyasa sakin |
        | **US10Y** | 💳 Borçlanma Maliyeti | Likidite azalıyor | Likidite artıyor |
        | **Gold** | 🥇 Güvenli Liman | Yatırımcılar korkuyor | Yatırımcılar risk alıyor |
        | **JPY** | 🇯🇵 Japonya Etkisi | Yen zayıf, carry trade | Yen güçlü, risk-off |
        
        **Özet**: Düşük DXY + Düşük VIX + Düşük faiz = **Risk-on ortam (kripto için iyi)**
        """)
    
    st.divider()
    
    # ==================== ⚡ ALTCOIN GÜÇ ENDEKSİ (BINANCE) ====================
    st.markdown("### ⚡ Altcoin Güç Endeksi")
    
    with st.spinner("Binance'den altcoin verileri alınıyor..."):
        altpower_score, btc_change = calculate_altpower_score()
    
    # Renk ve mesaj belirleme
    if altpower_score >= 60:
        bar_color = "#00C853"
        message = "🔥 ALTCOIN RALLİSİ: Altcoinler BTC'den daha güçlü!"
    elif altpower_score <= 30:
        bar_color = "#FF1744"
        message = "🛡️ BTC DOMİNASYONU: Altcoinler eziliyor."
    else:
        bar_color = "#FF9800"
        message = "⚖️ DENGELİ PİYASA"
    
    # Progress bar ve metrikler
    st.progress(altpower_score / 100)
    
    cols = st.columns([2, 1, 1])
    with cols[0]:
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; background: {bar_color}22; border-radius: 10px; border: 2px solid {bar_color};">
            <span style="color: {bar_color}; font-size: 1.3rem; font-weight: bold;">{message}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.metric(
            label="AltPower Skoru",
            value=f"{altpower_score:.0f}%",
            delta=f"{int(altpower_score/5)}/20 BTC'yi Geçti"
        )
    
    with cols[2]:
        st.metric(
            label="BTC 24H",
            value=f"{btc_change:+.2f}%",
            delta="Referans"
        )
    
    with st.expander("💡 Altcoin Güç Endeksi Nedir?"):
        st.markdown("""
        **AltPower Skoru**, piyasadaki 20 majör altcoinden kaçının son 24 saatte Bitcoin'den daha iyi performans gösterdiğini ölçer.
        
        | Skor | Durum | Anlam |
        |------|-------|-------|
        | ≥60% | 🔥 Altcoin Rallisi | Altcoinler BTC'den güçlü, altseason sinyali |
        | ≤30% | 🛡️ BTC Dominasyonu | Para BTC'ye akıyor, altcoinler zayıf |
        | 30-60% | ⚖️ Dengeli | Karışık piyasa, seçici olmak gerek |
        
        **Kaynak**: Binance (20 majör altcoin: ETH, BNB, SOL, XRP, ADA, DOGE, AVAX, TRX, DOT, MATIC, LTC, LINK, UNI, ATOM, ETC, FIL, NEAR, AAVE, QNT, ALGO)
        """)
    
    st.divider()
    
    # ==================== NAKİT AKIŞ TABLOSU ====================
    st.subheader("💸 Nakit Akışı Tablosu (Son 1 Saat)")
    
    with st.spinner("Hacim verileri yükleniyor..."):
        inout_data = calculate_inout_flow()
    
    if inout_data:
        df_flow = pd.DataFrame(inout_data)
        
        # Görüntüleme için sütunları formatla
        df_display = df_flow[['symbol', 'flow_pct', 'flow_type']].copy()
        df_display.columns = ['Coin', 'Akış %', 'Yön']
        df_display['Akış %'] = df_display['Akış %'].apply(lambda x: f"{x:+.1f}%")
        
        # Tablo stillemesi için renkli satırlar
        def highlight_flow(row):
            if row['Yön'] == 'BUY':
                return ['background-color: rgba(0, 200, 83, 0.2)'] * len(row)
            elif row['Yön'] == 'SELL':
                return ['background-color: rgba(255, 23, 68, 0.2)'] * len(row)
            return [''] * len(row)
        
        styled_df = df_display.style.apply(highlight_flow, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Nakit akış verisi yüklenemedi.")
    
    st.divider()
    
    # ==================== TRENDSTRING ANALİZİ ====================
    st.subheader("📊 TrendString Analizi (4H)")
    
    trend_cols = st.columns(2)
    
    with trend_cols[0]:
        btc_trend = calculate_trendstring('BTC/USDT')
        st.markdown(f"""
        <div style="text-align: center; padding: 15px; background: #1e1e1e; border-radius: 10px;">
            <h3 style="color: #FF9800; margin: 0;">₿ Bitcoin</h3>
            <p style="font-size: 2rem; margin: 10px 0; letter-spacing: 5px;">{btc_trend['visual']}</p>
            <p style="color: #888; margin: 0;">{btc_trend['trendstring']} ({btc_trend['bullish_count']}/5 Yükseliş)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with trend_cols[1]:
        eth_trend = calculate_trendstring('ETH/USDT')
        st.markdown(f"""
        <div style="text-align: center; padding: 15px; background: #1e1e1e; border-radius: 10px;">
            <h3 style="color: #627EEA; margin: 0;">Ξ Ethereum</h3>
            <p style="font-size: 2rem; margin: 10px 0; letter-spacing: 5px;">{eth_trend['visual']}</p>
            <p style="color: #888; margin: 0;">{eth_trend['trendstring']} ({eth_trend['bullish_count']}/5 Yükseliş)</p>
        </div>
        """, unsafe_allow_html=True)


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


def render_ai_page():
    """Yapay Zeka Tahmin Sayfası - XGBoost + SHAP"""
    st.title("🤖 Yapay Zeka Tahmin")
    st.caption("XGBoost modeli ile BTC fiyat yönü tahmini ve SHAP açıklanabilirlik")
    st.divider()
    
    # ==================== VERİ HAZIRLAMA ====================
    st.subheader("📊 Model Veri Seti")
    
    with st.spinner("Veri hazırlanıyor..."):
        try:
            
            # BTC verisini çek
            btc = yf.Ticker('BTC-USD')
            btc_hist = btc.history(period='2y')
            
            if btc_hist.empty or len(btc_hist) < 200:
                st.warning("⚠️ Eğitim için yeterli veri seti toplanıyor... Daha sonra tekrar deneyin.")
                return
            
            # Feature Engineering
            df = btc_hist[['Close', 'Volume', 'High', 'Low']].copy()
            df = df.astype('float32')  # Bellek optimizasyonu
            
            # ===== STATIONARITY: Değişim oranları =====
            df['returns'] = df['Close'].pct_change()
            df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['volume_pct'] = df['Volume'].pct_change()
            df['high_pct'] = df['High'].pct_change()
            df['low_pct'] = df['Low'].pct_change()
            
            # ===== VOLATILITY: ATR (Average True Range) =====
            df['tr1'] = df['High'] - df['Low']
            df['tr2'] = abs(df['High'] - df['Close'].shift(1))
            df['tr3'] = abs(df['Low'] - df['Close'].shift(1))
            df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            df['ATR_14'] = df['true_range'].rolling(window=14).mean()
            df['ATR_pct'] = df['ATR_14'] / df['Close']  # Normalize
            df = df.drop(['tr1', 'tr2', 'tr3', 'true_range'], axis=1)
            
            # ===== MOMENTUM: ROC (Rate of Change) =====
            df['ROC_5'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5) * 100
            df['ROC_10'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10) * 100
            df['ROC_20'] = (df['Close'] - df['Close'].shift(20)) / df['Close'].shift(20) * 100
            
            # ===== VWAP (Volume Weighted Average Price) =====
            df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
            df['vwap'] = (df['typical_price'] * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
            df['vwap_diff'] = (df['Close'] - df['vwap']) / df['vwap'] * 100  # VWAP'tan uzaklık
            df = df.drop(['typical_price', 'vwap'], axis=1)
            
            # ===== VOLATILITY =====
            df['volatility_10'] = df['returns'].rolling(window=10).std()
            df['volatility_20'] = df['returns'].rolling(window=20).std()
            df['volatility_ratio'] = df['volatility_10'] / df['volatility_20']
            
            # ===== RSI =====
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI_14'] = 100 - (100 / (1 + rs))
            df['RSI_normalized'] = (df['RSI_14'] - 50) / 50  # -1 to 1 arası
            
            # ===== EMA Sinyalleri =====
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
            df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
            df['ema_20_diff'] = (df['Close'] - df['EMA_20']) / df['EMA_20'] * 100
            df['ema_50_diff'] = (df['Close'] - df['EMA_50']) / df['EMA_50'] * 100
            df['ema_signal_20_50'] = (df['EMA_20'] > df['EMA_50']).astype(int)
            df['ema_signal_50_200'] = (df['EMA_50'] > df['EMA_200']).astype(int)
            df = df.drop(['EMA_20', 'EMA_50', 'EMA_200'], axis=1)
            
            # ===== LAG FEATURES =====
            for lag in [1, 2, 3]:
                df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
                df[f'volume_pct_lag_{lag}'] = df['volume_pct'].shift(lag)
                df[f'RSI_lag_{lag}'] = df['RSI_normalized'].shift(lag)
            
            # ===== MACRO LAG FEATURES =====
            if 'master_features_final' in st.session_state:
                macro_features = st.session_state['master_features_final']
                for key, value in macro_features.items():
                    df[f'macro_{key}'] = float(value)
                # DXY, VIX lag features
                if 'macro_dxy' in df.columns:
                    for lag in [1, 2, 3]:
                        df[f'macro_dxy_lag_{lag}'] = df['macro_dxy'].shift(lag)
                if 'macro_vix' in df.columns:
                    for lag in [1, 2, 3]:
                        df[f'macro_vix_lag_{lag}'] = df['macro_vix'].shift(lag)
            
            # ===== MULTI-CLASS TARGET =====
            df['future_return'] = df['Close'].shift(-5) / df['Close'] - 1
            return_std = df['future_return'].std()
            threshold = return_std * 0.5
            
            # Multi-class: -1 (Aşağı), 0 (Nötr), 1 (Yukarı)
            df['target_multi'] = 0  # Nötr
            df.loc[df['future_return'] > threshold, 'target_multi'] = 1  # Yukarı
            df.loc[df['future_return'] < -threshold, 'target_multi'] = -1  # Aşağı
            
            # Binary target (fallback)
            df['target'] = (df['future_return'] > 0).astype(int)
            
            # NaN temizliği
            df = df.dropna()
            
            # Feature listesi
            exclude_cols = ['Close', 'Volume', 'High', 'Low', 'future_return', 'target', 'target_multi']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            st.success(f"✅ {len(df)} satır veri hazırlandı ({len(feature_cols)} feature)")
            
            # Multi-class dağılımı
            target_dist = df['target_multi'].value_counts()
            st.caption(f"Target dağılımı: ⬆️ Yukarı: {target_dist.get(1, 0)}, ➡️ Nötr: {target_dist.get(0, 0)}, ⬇️ Aşağı: {target_dist.get(-1, 0)}")
            
        except Exception as e:
            st.error(f"Veri hazırlama hatası: {str(e)}")
            return
    
    st.divider()
    
    # ==================== MODEL EĞİTİMİ ====================
    st.subheader("🧠 XGBoost Model Eğitimi")
    
    # Cache'de model var mı kontrol et
    model_trained = 'xgb_model' in st.session_state and st.session_state.xgb_model is not None
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if model_trained:
            st.success("✅ Model hazır (cache'de)")
        else:
            st.info("Model henüz eğitilmemiş")
    
    with col2:
        train_button = st.button("🚀 Modeli Eğit", type="primary")
    
    if train_button or not model_trained:
        with st.spinner("Model eğitiliyor... (Bu işlem 30-60 saniye sürebilir)"):
            try:
                from xgboost import XGBClassifier
                from sklearn.model_selection import TimeSeriesSplit, cross_val_score
                from sklearn.preprocessing import StandardScaler
                from sklearn.metrics import accuracy_score
                
                # Feature ve target ayır
                X = df[feature_cols].astype('float32')
                y = df['target']
                
                # ===== StandardScaler normalizasyon =====
                scaler = StandardScaler()
                X_scaled = pd.DataFrame(
                    scaler.fit_transform(X),
                    columns=X.columns,
                    index=X.index
                )
                
                # inf/nan temizliği
                X_scaled = X_scaled.replace([np.inf, -np.inf], 0).fillna(0)
                
                # TimeSeriesSplit cross-validation (overfitting önleme)
                tscv = TimeSeriesSplit(n_splits=5)
                
                # XGBoost modeli
                model = XGBClassifier(
                    n_estimators=150,
                    max_depth=6,
                    learning_rate=0.05,
                    objective='binary:logistic',
                    eval_metric='logloss',
                    use_label_encoder=False,
                    random_state=42,
                    n_jobs=-1,
                    subsample=0.8,
                    colsample_bytree=0.8
                )
                
                # İlk eğitim (feature importance için)
                train_size = len(X_scaled) - 200
                X_train, X_test = X_scaled.iloc[:train_size], X_scaled.iloc[train_size:]
                y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
                
                model.fit(X_train, y_train)
                
                # ===== Feature Selection: En düşük %20'yi ele =====
                importances = model.feature_importances_
                importance_df = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                # En iyi %80'i seç
                n_keep = int(len(feature_cols) * 0.8)
                selected_features = importance_df.head(n_keep)['feature'].tolist()
                
                # Seçilen feature'larla tekrar eğit
                X_train_selected = X_train[selected_features]
                X_test_selected = X_test[selected_features]
                
                # Final model
                model_final = XGBClassifier(
                    n_estimators=150,
                    max_depth=6,
                    learning_rate=0.05,
                    objective='binary:logistic',
                    eval_metric='logloss',
                    use_label_encoder=False,
                    random_state=42,
                    n_jobs=-1,
                    subsample=0.8,
                    colsample_bytree=0.8
                )
                
                # Cross-validation skorları
                cv_scores = cross_val_score(model_final, X_scaled[selected_features], y, cv=tscv, scoring='accuracy')
                avg_cv_score = np.mean(cv_scores)
                
                model_final.fit(X_train_selected, y_train)
                
                # Test accuracy
                y_pred = model_final.predict(X_test_selected)
                test_accuracy = accuracy_score(y_test, y_pred)
                
                # Session state'e kaydet
                st.session_state.xgb_model = model_final
                st.session_state.xgb_features = selected_features
                st.session_state.xgb_scaler = scaler
                st.session_state.xgb_accuracy = test_accuracy
                st.session_state.xgb_cv_score = avg_cv_score
                st.session_state.xgb_X_test = X_test_selected
                st.session_state.xgb_last_row = X_scaled[selected_features].iloc[-1:]
                st.session_state.xgb_importance = importance_df
                
                st.success(f"✅ Model eğitildi!")
                st.write(f"**Feature Sayısı**: {len(feature_cols)} → {len(selected_features)} (en iyi %80)")
                st.write(f"**Cross-Validation (5-Fold)**: {avg_cv_score:.1%} ± {np.std(cv_scores):.1%}")
                st.write(f"**Test Accuracy**: {test_accuracy:.1%}")
                
            except ImportError:
                st.error("❌ XGBoost kütüphanesi yüklü değil. requirements.txt'i kontrol edin.")
                return
            except Exception as e:
                st.error(f"Model eğitim hatası: {str(e)}")
                return
    
    st.divider()
    
    # ==================== TAHMİN ====================
    if 'xgb_model' in st.session_state and st.session_state.xgb_model is not None:
        st.subheader("🎯 Güncel Tahmin")
        
        model = st.session_state.xgb_model
        last_row = st.session_state.xgb_last_row
        
        # Tahmin yap
        prediction = model.predict(last_row)[0]
        proba = model.predict_proba(last_row)[0]
        
        bull_prob = proba[1] * 100  # Yükseliş olasılığı
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Gauge Chart
            if bull_prob >= 60:
                color = "#00C853"
                signal = "📈 YÜKSELİŞ"
            elif bull_prob <= 40:
                color = "#FF1744"
                signal = "📉 DÜŞÜŞ"
            else:
                color = "#FF9800"
                signal = "➡️ NÖTR"
            
            # Gauge Chart (Plotly)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=bull_prob,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Yükseliş Olasılığı", 'font': {'size': 16, 'color': '#888'}},
                number={'suffix': "%", 'font': {'size': 40, 'color': color}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#555"},
                    'bar': {'color': color},
                    'bgcolor': "#1e1e1e",
                    'borderwidth': 2,
                    'bordercolor': "#333",
                    'steps': [
                        {'range': [0, 40], 'color': 'rgba(255, 23, 68, 0.13)'},
                        {'range': [40, 60], 'color': 'rgba(255, 152, 0, 0.13)'},
                        {'range': [60, 100], 'color': 'rgba(0, 200, 83, 0.13)'}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': bull_prob
                    }
                }
            ))
            
            fig_gauge.update_layout(
                template="plotly_dark",
                height=250,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown(f"<h3 style='text-align:center; color:{color};'>{signal}</h3>", unsafe_allow_html=True)
            st.caption(f"Model Accuracy: {st.session_state.xgb_accuracy:.1%}")
        
        with col2:
            # Basitleştirilmiş SHAP - İnsan okunabilir
            st.write("**🎯 Tahmini Etkileyen En Önemli 3 Faktör**")
            
            # Feature isimlerini insan dostu hale getir
            feature_labels = {
                'returns': '📈 Fiyat Değişimi',
                'RSI_14': '📊 RSI (Aşırı alım/satım)',
                'RSI_normalized': '📊 RSI Durumu',
                'volatility_20': '🌊 Volatilite',
                'volatility_10': '🌊 Kısa Vadeli Volatilite',
                'ROC_5': '🚀 Kısa Momentum',
                'ROC_10': '🚀 Orta Momentum',
                'ROC_20': '🚀 Uzun Momentum',
                'ATR_pct': '📏 ATR (Volatilite)',
                'ema_20_diff': '📉 EMA-20 Uzaklığı',
                'ema_50_diff': '📉 EMA-50 Uzaklığı',
                'vwap_diff': '💰 VWAP Farkı',
                'volume_pct': '📊 Hacim Değişimi',
                'ema_signal_20_50': '🚦 EMA Kesişimi',
                'macro_dxy': '💵 Doların Gücü',
                'macro_vix': '😱 Korku Endeksi',
                'log_returns': '📈 Log Getiri'
            }
            
            if 'xgb_importance' in st.session_state:
                top3 = st.session_state.xgb_importance.head(3)
                
                for i, row in top3.iterrows():
                    feat_name = row['feature']
                    human_name = feature_labels.get(feat_name, feat_name)
                    importance = row['importance']
                    
                    # Renk belirle
                    if i == 0:
                        rank_color = "#FFD700"  # Altın
                        rank_icon = "🥇"
                    elif i == 1:
                        rank_color = "#C0C0C0"  # Gümüş
                        rank_icon = "🥈"
                    else:
                        rank_color = "#CD7F32"  # Bronz
                        rank_icon = "🥉"
                    
                    st.markdown(f"""
                    <div style="padding: 12px; background: #2a2a2a; border-radius: 10px; margin-bottom: 10px; border-left: 4px solid {rank_color};">
                        <span style="font-size: 1.2rem;">{rank_icon}</span>
                        <span style="color: #fff; font-weight: bold;"> {human_name}</span>
                        <span style="color: #888; float: right;">({importance:.3f})</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Feature importance hesaplanmadı")
        
        st.divider()
        
        # Model detayları
        with st.expander("🔍 Model Detayları"):
            st.write(f"**Eğitim Veri Boyutu**: {len(df) - 200} satır")
            st.write(f"**Test Veri Boyutu**: 200 satır")
            st.write(f"**Feature Sayısı**: {len(st.session_state.xgb_features)}")
            st.write(f"**Target**: 5 periyot sonraki yön (0: Düşüş, 1: Yükseliş)")
            
            st.divider()
            st.write("**Kullanılan Features:**")
            st.write(", ".join(st.session_state.xgb_features[:15]) + "...")
        
        # Uyarı
        st.warning("⚠️ Bu tahminler yalnızca bilgilendirme amaçlıdır ve yatırım tavsiyesi değildir. Model geçmiş verilerle eğitilmiştir ve gelecek performansı garanti etmez.")
    
    else:
        st.info("Tahmin yapmak için önce modeli eğitin.")


# ==================== BACKTEST ENGINE ====================

def run_backtest(predictions, prices, fee: float = 0.001) -> dict:
    """
    Vectorized Backtest Engine.
    
    Args:
        predictions: Model tahminleri (0: Sat, 1: Al)
        prices: Fiyat serisi
        fee: İşlem başına komisyon (default: %0.1)
    
    Returns:
        dict: Backtest sonuçları
    """
    
    # Array'leri aynı boyuta getir
    min_len = min(len(predictions), len(prices))
    predictions = predictions[:min_len]
    prices = prices[:min_len]
    
    # Getiriler (n-1 uzunlukta)
    returns = np.diff(prices) / prices[:-1]
    
    # Predictions'ı returns ile aynı boyuta getir
    pred_aligned = predictions[:-1]
    
    # Sinyal değişimlerini bul (alım-satım noktaları)
    signal_changes = np.diff(pred_aligned)
    trades = np.sum(np.abs(signal_changes))
    
    # Strateji getirileri (sinyal 1 ise long, 0 ise cash)
    strategy_returns = pred_aligned * returns
    
    # Komisyon maliyeti (her işlemde) - sigmoid_changes 1 eksik
    if len(signal_changes) > 0:
        trade_costs = np.zeros_like(strategy_returns)
        trade_costs[1:] = np.abs(signal_changes) * fee
        strategy_returns = strategy_returns - trade_costs
    
    # Kümülatif getiriler
    cumulative_strategy = np.cumprod(1 + strategy_returns) - 1
    cumulative_buyhold = np.cumprod(1 + returns) - 1
    
    # Toplam getiriler
    total_strategy_return = cumulative_strategy[-1] * 100 if len(cumulative_strategy) > 0 else 0
    total_buyhold_return = cumulative_buyhold[-1] * 100 if len(cumulative_buyhold) > 0 else 0
    
    # Sharpe Ratio (yıllıklandırılmış, risk-free rate = 0)
    daily_std = np.std(strategy_returns)
    if daily_std > 0:
        sharpe_ratio = (np.mean(strategy_returns) / daily_std) * np.sqrt(252)
    else:
        sharpe_ratio = 0
    
    # Max Drawdown
    cumulative_wealth = np.cumprod(1 + strategy_returns)
    peak = np.maximum.accumulate(cumulative_wealth)
    drawdown = (peak - cumulative_wealth) / peak
    max_drawdown = np.max(drawdown) * 100
    
    # Win Rate
    winning_trades = np.sum(strategy_returns > 0)
    total_trades = np.sum(strategy_returns != 0)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Sortino Ratio (sadece negatif volatilite)
    negative_returns = strategy_returns[strategy_returns < 0]
    downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 0
    if downside_std > 0:
        sortino_ratio = (np.mean(strategy_returns) / downside_std) * np.sqrt(252)
    else:
        sortino_ratio = 0
    
    # Recovery Factor (toplam getiri / max drawdown)
    if max_drawdown > 0:
        recovery_factor = total_strategy_return / max_drawdown
    else:
        recovery_factor = float('inf') if total_strategy_return > 0 else 0
    
    # Calmar Ratio (yıllık getiri / max drawdown)
    annual_return = total_strategy_return  # Basitleştirilmiş
    calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
    
    return {
        'strategy_returns': strategy_returns,
        'cumulative_strategy': cumulative_strategy,
        'cumulative_buyhold': cumulative_buyhold,
        'total_strategy_return': total_strategy_return,
        'total_buyhold_return': total_buyhold_return,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'recovery_factor': recovery_factor,
        'calmar_ratio': calmar_ratio,
        'total_trades': int(trades),
        'win_rate': win_rate,
        'total_fees': trades * fee * 100
    }


@st.cache_data(ttl=86400, show_spinner=False)  # 1 günlük cache
def fetch_backtest_data(symbol: str = 'BTC-USD', period: str = '2y'):
    """Backtest için tarihsel veri çeker."""
    
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        
        if hist.empty:
            return None, "Veri alınamadı"
        
        return hist, None
    except Exception as e:
        return None, str(e)


def render_backtest_page():
    """Backtest Sayfası - Strateji Performans Testi"""
    st.title("📈 Backtest - Strateji Performans")
    st.caption("XGBoost tahminlerini geçmiş veriler üzerinde test edin")
    st.divider()
    
    # Model kontrolü
    if 'xgb_model' not in st.session_state or st.session_state.xgb_model is None:
        st.warning("⚠️ Önce 🤖 AI Tahmin sayfasından modeli eğitin.")
        st.info("Model eğitildikten sonra bu sayfada backtest yapabilirsiniz.")
        return
    
    model = st.session_state.xgb_model
    feature_cols = st.session_state.xgb_features
    
    st.success(f"✅ Model hazır (Accuracy: {st.session_state.xgb_accuracy:.1%})")
    
    st.divider()
    
    # ==================== VERİ HAZIRLAMA ====================
    st.subheader("📊 Backtest Veri Seti")
    
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.selectbox("Sembol", ['BTC-USD', 'ETH-USD'], index=0)
    
    with col2:
        period = st.selectbox("Dönem", ['1y', '2y', '5y'], index=1)
    
    with st.spinner("Veri hazırlanıyor..."):
        hist, error = fetch_backtest_data(symbol, period)
        
        if error:
            st.error(f"Veri hatası: {error}")
            return
        
        try:
            
            # Feature Engineering (AI sayfasıyla aynı)
            df = hist[['Close', 'Volume', 'High', 'Low']].copy()
            df = df.astype('float32')
            
            df['returns'] = df['Close'].pct_change()
            df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['volatility_20'] = df['returns'].rolling(window=20).std()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI_14'] = 100 - (100 / (1 + rs))
            
            # EMA
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
            df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
            
            df['ema_signal_20_50'] = (df['EMA_20'] > df['EMA_50']).astype(int)
            df['ema_signal_50_200'] = (df['EMA_50'] > df['EMA_200']).astype(int)
            
            df['momentum_5'] = df['Close'].pct_change(5)
            df['momentum_10'] = df['Close'].pct_change(10)
            df['momentum_20'] = df['Close'].pct_change(20)
            
            df['high_low_ratio'] = df['High'] / df['Low']
            df['volume_change'] = df['Volume'].pct_change()
            
            # Makro features (varsa)
            if 'master_features_final' in st.session_state:
                for key, value in st.session_state['master_features_final'].items():
                    df[f'macro_{key}'] = float(value)
            
            df = df.dropna()
            
            # Feature'ları kontrol et
            available_features = [col for col in feature_cols if col in df.columns]
            missing_features = [col for col in feature_cols if col not in df.columns]
            
            if len(available_features) < len(feature_cols) * 0.5:
                st.error("Yeterli feature bulunamadı. Model uyumsuz.")
                return
            
            # Eksik feature'lara 0 ata
            for feat in missing_features:
                df[feat] = 0.0
            
            X = df[feature_cols].astype('float32')
            prices = df['Close'].values
            
            st.success(f"✅ {len(df)} gün veri hazırlandı")
            
        except Exception as e:
            st.error(f"Veri hazırlama hatası: {str(e)}")
            return
    
    st.divider()
    
    # ==================== BACKTEST ====================
    st.subheader("🚀 Backtest Çalıştır")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fee = st.slider("İşlem Komisyonu (%)", 0.0, 0.5, 0.1, 0.05) / 100
    
    with col2:
        run_button = st.button("📊 Backtest Başlat", type="primary")
    
    if run_button:
        with st.spinner("Backtest çalıştırılıyor..."):
            try:
                # Tahminleri üret
                predictions = model.predict(X)
                
                # Backtest çalıştır
                results = run_backtest(predictions, prices, fee)
                
                # Session state'e kaydet
                st.session_state.backtest_results = results
                st.session_state.backtest_dates = df.index[:-1]  # returns 1 eksik
                
                st.success("✅ Backtest tamamlandı!")
                
            except Exception as e:
                st.error(f"Backtest hatası: {str(e)}")
                return
    
    # ==================== SONUÇLAR ====================
    if 'backtest_results' in st.session_state:
        results = st.session_state.backtest_results
        dates = st.session_state.backtest_dates
        
        st.divider()
        st.subheader("📊 Performans Sonuçları")
        
        # Metrik kartları
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            strat_color = "#00C853" if results['total_strategy_return'] > 0 else "#FF1744"
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; background: {strat_color}22; border-radius: 10px; border: 2px solid {strat_color};">
                <p style="margin: 0; color: #888;">📈 Strateji Getirisi</p>
                <h2 style="color: {strat_color}; margin: 5px 0;">{results['total_strategy_return']:+.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_cols[1]:
            bh_color = "#00C853" if results['total_buyhold_return'] > 0 else "#FF1744"
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; background: {bh_color}22; border-radius: 10px; border: 2px solid {bh_color};">
                <p style="margin: 0; color: #888;">📊 Al-Tut Getirisi</p>
                <h2 style="color: {bh_color}; margin: 5px 0;">{results['total_buyhold_return']:+.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_cols[2]:
            sharpe_color = "#00C853" if results['sharpe_ratio'] > 1 else "#FF9800" if results['sharpe_ratio'] > 0 else "#FF1744"
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; background: {sharpe_color}22; border-radius: 10px; border: 2px solid {sharpe_color};">
                <p style="margin: 0; color: #888;">📐 Sharpe Ratio</p>
                <h2 style="color: {sharpe_color}; margin: 5px 0;">{results['sharpe_ratio']:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_cols[3]:
            dd_color = "#00C853" if results['max_drawdown'] < 20 else "#FF9800" if results['max_drawdown'] < 40 else "#FF1744"
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; background: {dd_color}22; border-radius: 10px; border: 2px solid {dd_color};">
                <p style="margin: 0; color: #888;">📉 Max Drawdown</p>
                <h2 style="color: {dd_color}; margin: 5px 0;">{results['max_drawdown']:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # İkinci satır metrikler (yeni metrikler varsa göster)
        sortino = results.get('sortino_ratio', 0)
        recovery = results.get('recovery_factor', 0)
        calmar = results.get('calmar_ratio', 0)
        win_rate = results.get('win_rate', 0)
        
        metric_cols2 = st.columns(4)
        
        with metric_cols2[0]:
            sortino_color = "#00C853" if sortino > 1.5 else "#FF9800" if sortino > 0 else "#FF1744"
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; background: {sortino_color}22; border-radius: 10px; border: 2px solid {sortino_color};">
                <p style="margin: 0; color: #888;">📊 Sortino Ratio</p>
                <h2 style="color: {sortino_color}; margin: 5px 0;">{sortino:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_cols2[1]:
            rf_display = f"{recovery:.2f}" if recovery != float('inf') else "∞"
            rf_color = "#00C853" if recovery > 2 else "#FF9800" if recovery > 1 else "#FF1744"
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; background: {rf_color}22; border-radius: 10px; border: 2px solid {rf_color};">
                <p style="margin: 0; color: #888;">🔄 Recovery Factor</p>
                <h2 style="color: {rf_color}; margin: 5px 0;">{rf_display}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_cols2[2]:
            calmar_color = "#00C853" if calmar > 1 else "#FF9800" if calmar > 0 else "#FF1744"
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; background: {calmar_color}22; border-radius: 10px; border: 2px solid {calmar_color};">
                <p style="margin: 0; color: #888;">📈 Calmar Ratio</p>
                <h2 style="color: {calmar_color}; margin: 5px 0;">{calmar:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_cols2[3]:
            wr_color = "#00C853" if win_rate > 55 else "#FF9800" if win_rate > 45 else "#FF1744"
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; background: {wr_color}22; border-radius: 10px; border: 2px solid {wr_color};">
                <p style="margin: 0; color: #888;">🎯 Win Rate</p>
                <h2 style="color: {wr_color}; margin: 5px 0;">{win_rate:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Equity Curve
        st.subheader("📈 Equity Curve")
        
        fig = go.Figure()
        
        # Strateji
        fig.add_trace(go.Scatter(
            x=dates,
            y=results['cumulative_strategy'] * 100,
            name='XGBoost Strateji',
            line=dict(color='#2196F3', width=2)
        ))
        
        # Buy & Hold
        fig.add_trace(go.Scatter(
            x=dates,
            y=results['cumulative_buyhold'] * 100,
            name='Al-Tut (Buy & Hold)',
            line=dict(color='#FF9800', width=2)
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            template="plotly_dark",
            height=400,
            margin=dict(l=0, r=0, t=30, b=20),
            yaxis_title="Kümülatif Getiri (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detaylı istatistikler
        with st.expander("📋 Detaylı İstatistikler"):
            stat_cols = st.columns(3)
            
            with stat_cols[0]:
                st.metric("Toplam İşlem", f"{results['total_trades']}")
                st.metric("Win Rate", f"{results['win_rate']:.1f}%")
            
            with stat_cols[1]:
                st.metric("Toplam Komisyon", f"{results['total_fees']:.2f}%")
                st.metric("Net Getiri", f"{results['total_strategy_return'] - results['total_fees']:.1f}%")
            
            with stat_cols[2]:
                excess_return = results['total_strategy_return'] - results['total_buyhold_return']
                st.metric("Alpha (Aşırı Getiri)", f"{excess_return:+.1f}%")
        
        # Uyarı
        st.warning("⚠️ Geçmiş performans gelecek sonuçları garanti etmez. Bu backtest simülasyonu yalnızca bilgilendirme amaçlıdır.")


# ==================== V2.0 YENİ SAYFA FONKSİYONLARI ====================

def calculate_fft_cycles(prices):
    """FFT ile fiyat döngülerini tespit eder."""
    try:
        # Trend kaldır
        x = np.arange(len(prices))
        coeffs = np.polyfit(x, prices, 1)
        trend = np.polyval(coeffs, x)
        detrended = prices - trend
        
        # FFT hesapla
        n = len(detrended)
        yf = fft(detrended)
        xf = fftfreq(n, 1)
        
        # Pozitif frekanslar ve güç
        pos_mask = xf > 0
        freqs = xf[pos_mask]
        power = np.abs(yf[pos_mask])
        
        # Dominant period
        if len(power) > 0:
            dominant_idx = np.argmax(power)
            dominant_period = 1 / freqs[dominant_idx] if freqs[dominant_idx] > 0 else 0
        else:
            dominant_period = 0
        
        # Top 5 döngü
        top_indices = np.argsort(power)[-5:][::-1]
        top_cycles = [(1/freqs[i] if freqs[i] > 0 else 0, power[i]) for i in top_indices if freqs[i] > 0]
        
        return {
            'dominant_period': dominant_period,
            'frequencies': freqs,
            'power': power,
            'top_cycles': top_cycles
        }
    except Exception as e:
        return {'dominant_period': 0, 'frequencies': [], 'power': [], 'top_cycles': [], 'error': str(e)}


def calculate_kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> dict:
    """Kelly Criterion ile optimal pozisyon boyutu."""
    if avg_loss == 0 or win_rate == 0:
        return {'kelly_full': 0, 'kelly_half': 0, 'recommendation': 'Yetersiz veri'}
    
    win_loss_ratio = avg_win / abs(avg_loss)
    kelly_full = win_rate - ((1 - win_rate) / win_loss_ratio)
    kelly_half = kelly_full / 2
    
    if kelly_full <= 0:
        recommendation = "❌ Bu strateji ile yatırım yapılmamalı"
    elif kelly_full < 0.1:
        recommendation = "⚠️ Çok küçük pozisyon (<%10)"
    elif kelly_full < 0.25:
        recommendation = "✅ Makul pozisyon boyutu"
    else:
        recommendation = "🔥 Agresif (Half-Kelly önerilir)"
    
    return {
        'kelly_full': max(0, kelly_full) * 100,
        'kelly_half': max(0, kelly_half) * 100,
        'recommendation': recommendation
    }


def render_kokpit():
    """🏠 KOKPİT - Executive Dashboard"""
    st.title("🏠 KOKPİT")
    st.caption("Tek bakışta piyasa durumu ve yatırım kararı")
    
    # Karar Kutusu (mevcut Dashboard'dan)
    ai_prob = None
    risk_score = st.session_state.get('risk_score', 50)
    market_regime = st.session_state.get('market_regime', 'KARIŞIK')
    
    if 'xgb_model' in st.session_state and st.session_state.xgb_model is not None:
        try:
            last_row = st.session_state.xgb_last_row
            proba = st.session_state.xgb_model.predict_proba(last_row)[0]
            ai_prob = proba[1] * 100
        except:
            ai_prob = None
    
    # Executive Summary Box
    if ai_prob is not None and ai_prob > 55 and risk_score > 60:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(0,200,83,0.13), rgba(0,200,83,0.27)); border: 3px solid #00C853; border-radius: 15px; padding: 25px; margin-bottom: 20px;">
            <h2 style="color: #00C853; margin: 0; text-align: center;">✅ YATIRIM İÇİN UYGUN KOŞULLAR</h2>
            <p style="color: #888; text-align: center; margin: 10px 0;">AI tahmini olumlu, makro riskler düşük.</p>
        </div>
        """, unsafe_allow_html=True)
    elif ai_prob is not None and ai_prob < 45 or risk_score < 40:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(255,23,68,0.13), rgba(255,23,68,0.27)); border: 3px solid #FF1744; border-radius: 15px; padding: 25px; margin-bottom: 20px;">
            <h2 style="color: #FF1744; margin: 0; text-align: center;">⚠️ RİSK YÜKSEK - KORUNMA MODU</h2>
            <p style="color: #888; text-align: center; margin: 10px 0;">Dikkatli olun, nakit/altın pozisyonu düşünün.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(255,152,0,0.13), rgba(255,152,0,0.27)); border: 3px solid #FF9800; border-radius: 15px; padding: 25px; margin-bottom: 20px;">
            <h2 style="color: #FF9800; margin: 0; text-align: center;">🔄 KARIŞIK SİNYALLER</h2>
            <p style="color: #888; text-align: center; margin: 10px 0;">Küçük pozisyonlar, stop-loss kullanın.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # 3 Kritik Metrik
    st.subheader("📊 Kritik Metrikler")
    cols = st.columns(3)
    
    with cols[0]:
        if ai_prob is not None:
            ai_color = "#00C853" if ai_prob > 55 else "#FF1744" if ai_prob < 45 else "#FF9800"
            st.metric("🤖 AI Puanı", f"{ai_prob:.0f}%")
        else:
            st.metric("🤖 AI Puanı", "Model eğitilmedi")
    
    with cols[1]:
        risk_color = "#00C853" if risk_score > 60 else "#FF1744" if risk_score < 40 else "#FF9800"
        st.metric("🧭 Makro Risk", f"{risk_score:.0f}/100")
    
    with cols[2]:
        btc_data, _, _ = fetch_crypto_ticker("BTC/USDT")
        if btc_data:
            btc_price = btc_data.get('last', 0)
            btc_change = btc_data.get('percentage', 0)
            st.metric("₿ BTC Fiyatı", f"${btc_price:,.0f}", f"{btc_change:+.2f}%")
        else:
            st.metric("₿ BTC Fiyatı", "—")
    
    st.divider()
    
    # AltPower Bar
    st.subheader("⚡ Altcoin Güç Endeksi")
    with st.spinner("Altcoin verileri yükleniyor..."):
        altpower_score, btc_change = calculate_altpower_score()
    
    st.progress(altpower_score / 100)
    
    if altpower_score >= 60:
        st.success(f"🔥 ALTCOIN RALLİSİ: {altpower_score:.0f}% altcoin BTC'den güçlü")
    elif altpower_score <= 30:
        st.error(f"🛡️ BTC DOMİNASYONU: Sadece {altpower_score:.0f}% altcoin BTC'yi geçiyor")
    else:
        st.warning(f"⚖️ DENGELİ: {altpower_score:.0f}% altcoin BTC'den iyi")


def render_piyasa_radari():
    """📡 PİYASA RADARI - Tüm Mikabot Özellikleri"""
    st.title("📡 PİYASA RADARI")
    st.caption("Kripto piyasası anlık tarama ve analiz merkezi")
    
    tabs = st.tabs(["📊 TrendString", "💸 InOut Akış", "🔥 SVI Sıkışma", "📚 Orderbook", "📐 Channel Bender"])
    
    # TrendString Tab
    with tabs[0]:
        st.markdown("#### 📊 TrendString Tablosu")
        st.caption("Top 10 coin için son 5 adet 4H mumun yönü")
        
        with st.spinner("Piyasa radarı yükleniyor..."):
            radar_data = fetch_market_radar_data()
        
        if radar_data:
            df_radar = pd.DataFrame(radar_data)
            df_view = df_radar[['Coin', 'Fiyat', 'TrendString', 'InOut', '24s Değişim']].copy()
            df_view.columns = ['Coin', 'Fiyat ($)', 'Trend (4H)', 'Nakit Akış', '24H (%)']
            st.dataframe(df_view, use_container_width=True, hide_index=True)
        else:
            st.warning("Veri yüklenemedi")
    
    # InOut Tab
    with tabs[1]:
        st.markdown("#### 💸 Nakit Akışı (Son 1 Saat)")
        with st.spinner("Hacim verileri yükleniyor..."):
            inout_data = calculate_inout_flow()
        
        if inout_data:
            df_flow = pd.DataFrame(inout_data)
            st.dataframe(df_flow[['symbol', 'flow_pct', 'flow_type']], use_container_width=True, hide_index=True)
        else:
            st.warning("Veri yüklenemedi")
    
    # SVI Tab
    with tabs[2]:
        st.markdown("#### 🔥 Volatilite Sıkışması (Bollinger Bandwidth)")
        with st.spinner("Sıkışma analizi..."):
            squeeze_data = calculate_squeeze_volatility()
        
        if squeeze_data:
            df_sq = pd.DataFrame(squeeze_data)
            alerts = [s for s in squeeze_data if s['SqueezeAlert']]
            if alerts:
                st.warning(f"⚠️ {len(alerts)} coin sıkışma bölgesinde!")
            st.dataframe(df_sq[['Coin', 'Bandwidth', 'SqueezeStatus']], use_container_width=True, hide_index=True)
        else:
            st.warning("Veri yüklenemedi")
    
    # Orderbook Tab
    with tabs[3]:
        st.markdown("#### 📚 Emir Defteri Dengesizliği")
        with st.spinner("Orderbook verileri..."):
            ob_data = calculate_orderbook_imbalance()
        
        if ob_data:
            df_ob = pd.DataFrame(ob_data)
            st.dataframe(df_ob[['Coin', 'Imbalance', 'Status']], use_container_width=True, hide_index=True)
        else:
            st.warning("Veri yüklenemedi")
    
    # Channel Bender Tab
    with tabs[4]:
        st.markdown("#### 📐 Kanal Bükücü (Bollinger Sapma)")
        with st.spinner("Kanal analizi..."):
            ch_data = calculate_channel_bender()
        
        if ch_data:
            df_ch = pd.DataFrame(ch_data)
            st.dataframe(df_ch[['Coin', 'Price', 'DeviationScore', 'Status']], use_container_width=True, hide_index=True)
        else:
            st.warning("Veri yüklenemedi")


def render_quant_lab():
    """🧠 QUANT LABORATUVARI - Gelişmiş Analiz"""
    st.title("🧠 QUANT LABORATUVARI")
    st.caption("Yapay zeka ve istatistiksel analiz merkezi")
    
    tabs = st.tabs(["🤖 XGBoost Tahmin", "📊 SHAP Analizi", "🌊 FFT Döngü", "🎰 Kelly Hesaplayıcı"])
    
    # XGBoost Tab - mevcut render_ai_page içeriği
    with tabs[0]:
        render_ai_page()
    
    # SHAP Tab
    with tabs[1]:
        st.markdown("#### 📊 SHAP Feature Importance")
        if 'xgb_importance' in st.session_state:
            importance_df = st.session_state.xgb_importance.head(10)
            fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                        color='importance', color_continuous_scale='Viridis')
            fig.update_layout(template='plotly_dark', height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Önce XGBoost modelini eğitin.")
    
    # FFT Tab
    with tabs[2]:
        st.markdown("#### 🌊 FFT Döngü Analizi")
        st.caption("Fiyat serisindeki dominant döngüleri tespit eder")
        
        with st.spinner("FFT hesaplanıyor..."):
            try:
                btc = yf.Ticker('BTC-USD')
                btc_hist = btc.history(period='1y')
                if not btc_hist.empty:
                    prices = btc_hist['Close'].values
                    fft_result = calculate_fft_cycles(prices)
                    
                    st.metric("⏰ Dominant Döngü", f"{fft_result['dominant_period']:.0f} gün")
                    
                    if fft_result.get('top_cycles'):
                        st.markdown("**Top 5 Döngü:**")
                        for i, (period, power) in enumerate(fft_result['top_cycles'][:5]):
                            if period > 0:
                                st.write(f"{i+1}. {period:.0f} gün (güç: {power:.0f})")
                    
                    # FFT grafiği
                    if len(fft_result['frequencies']) > 0:
                        fig = go.Figure()
                        periods = 1 / fft_result['frequencies']
                        mask = (periods > 5) & (periods < 200)
                        fig.add_trace(go.Scatter(x=periods[mask], y=fft_result['power'][mask], mode='lines', fill='tozeroy'))
                        fig.update_layout(template='plotly_dark', xaxis_title='Periyot (gün)', yaxis_title='Güç', height=300)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("BTC verisi alınamadı")
            except Exception as e:
                st.error(f"FFT hatası: {str(e)}")
    
    # Kelly Tab
    with tabs[3]:
        st.markdown("#### 🎰 Kelly Criterion Hesaplayıcı")
        st.caption("Optimal pozisyon boyutu hesaplama")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            win_rate = st.slider("Win Rate (%)", 0, 100, 55) / 100
        with col2:
            avg_win = st.number_input("Ortalama Kazanç (%)", value=3.0)
        with col3:
            avg_loss = st.number_input("Ortalama Kayıp (%)", value=2.0)
        
        if st.button("Hesapla"):
            kelly = calculate_kelly_fraction(win_rate, avg_win, avg_loss)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Full Kelly", f"{kelly['kelly_full']:.1f}%")
            with col2:
                st.metric("Half Kelly (Önerilen)", f"{kelly['kelly_half']:.1f}%")
            
            st.info(kelly['recommendation'])


def render_makro_temel():
    """🌍 MAKRO & TEMEL - Ekonomi ve On-Chain"""
    st.title("🌍 MAKRO & TEMEL ANALİZ")
    st.caption("Küresel ekonomi ve blockchain temel verileri")
    
    tabs = st.tabs(["💵 DXY & Faizler", "⛓️ On-Chain (TVL)", "📰 Sentiment", "📈 Hisse Piyasası"])
    
    # DXY Tab - mevcut makro sayfasından
    with tabs[0]:
        render_macro_page()
    
    # On-Chain Tab
    with tabs[1]:
        render_onchain_page()
    
    # Sentiment Tab
    with tabs[2]:
        st.markdown("#### 📰 Piyasa Duyarlılığı")
        with st.spinner("Sentiment verileri yükleniyor..."):
            fng_data, _ = fetch_fear_greed_index()
        
        if fng_data:
            fng_val = fng_data['value']
            if fng_val < 25:
                fng_color, fng_label = "#FF1744", "Extreme Fear"
            elif fng_val < 45:
                fng_color, fng_label = "#FF5722", "Fear"
            elif fng_val < 55:
                fng_color, fng_label = "#FF9800", "Neutral"
            elif fng_val < 75:
                fng_color, fng_label = "#8BC34A", "Greed"
            else:
                fng_color, fng_label = "#00C853", "Extreme Greed"
            
            st.metric("😱 Fear & Greed Index", f"{fng_val} - {fng_label}")
            st.progress(fng_val / 100)
        else:
            st.warning("Sentiment verisi alınamadı")
    
    # Hisse Tab
    with tabs[3]:
        render_stock_page()


def render_sistem():
    """⚙️ SİSTEM - Backtest ve Ayarlar"""
    st.title("⚙️ SİSTEM")
    st.caption("Strateji testi ve uygulama ayarları")
    
    tabs = st.tabs(["📉 Backtest", "🔧 Ayarlar"])
    
    with tabs[0]:
        render_backtest_page()
    
    with tabs[1]:
        render_settings_page()


def render_sidebar():
    """Sidebar navigasyon - v2.0 Profesyonel Hiyerarşi"""
    st.sidebar.title("📊 Finans Terminali")
    st.sidebar.caption("v2.0 Profesyonel")
    st.sidebar.divider()
    
    pages = [
        '🏠 KOKPİT',
        '📡 PİYASA RADARI',
        '🧠 QUANT LABORATUVARI',
        '🌍 MAKRO & TEMEL',
        '⚙️ SİSTEM'
    ]
    
    selected = st.sidebar.radio("Menü", pages, label_visibility="collapsed")
    
    st.sidebar.divider()
    st.sidebar.caption("💡 Veriler önbelleğe alınır")
    st.sidebar.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')}")
    
    return selected


# ==================== ANA ROUTER ====================

def main():
    """Ana uygulama - v2.0 Router"""
    selected_page = render_sidebar()
    
    if selected_page == '🏠 KOKPİT':
        render_kokpit()
    elif selected_page == '📡 PİYASA RADARI':
        render_piyasa_radari()
    elif selected_page == '🧠 QUANT LABORATUVARI':
        render_quant_lab()
    elif selected_page == '🌍 MAKRO & TEMEL':
        render_makro_temel()
    elif selected_page == '⚙️ SİSTEM':
        render_sistem()
    
    # Footer
    st.divider()
    st.caption("📊 Finans Terminali v2.0 | Veriler bilgilendirme amaçlıdır, yatırım tavsiyesi değildir.")


if __name__ == "__main__":
    main()

