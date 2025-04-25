import pandas as pd
import numpy as np
import talib
from ta import add_all_ta_features
from typing import Dict, Tuple
import ccxt
from datetime import datetime
import requests  # Makro veriler için

class DataProcessor:
    def __init__(self, exchange: ccxt.Exchange):
        self.exchange = exchange
        self.macro_data_cache = {}
        
    def fetch_and_preprocess(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        """Binance'dan veri çekip ön işleme yapar"""
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = self._convert_to_dataframe(ohlcv)
        df = self._add_technical_indicators(df)
        df = self._add_orderbook_features(symbol, df)
        df = self._add_macro_features(df)
        df = self._clean_data(df)
        return df

    def _convert_to_dataframe(self, ohlcv: list) -> pd.DataFrame:
        """CCXT verisini DataFrame'e çevirir"""
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """TA-Lib ve özel göstergeler ekler"""
        # TA-Lib göstergeleri
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['macd'], df['macd_signal'], _ = talib.MACD(df['close'])
        
        # Volume Profile
        df['vp_buy'] = df['volume'] * (df['close'] - df['low']) / (df['high'] - df['low'])
        df['vp_sell'] = df['volume'] - df['vp_buy']
        
        # Tüm TA-Lib göstergeleri (ta kütüphanesi ile)
        df = add_all_ta_features(
            df, 
            open="open", high="high", low="low", 
            close="close", volume="volume"
        )
        
        return df

    def _add_orderbook_features(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """Order book derinlik analizi ekler"""
        orderbook = self.exchange.fetch_order_book(symbol)
        
        # Derinlik analizi
        for depth in [0.001, 0.005, 0.01]:  # %0.1, %0.5, %1 seviyeleri
            price_offset = df['close'].iloc[-1] * depth
            bid_vol = sum([x[1] for x in orderbook['bids'] if x[0] >= df['close'].iloc[-1] - price_offset])
            ask_vol = sum([x[1] for x in orderbook['asks'] if x[0] <= df['close'].iloc[-1] + price_offset])
            
            df[f'bid_depth_{depth}'] = bid_vol
            df[f'ask_depth_{depth}'] = ask_vol
            df[f'depth_ratio_{depth}'] = bid_vol / (ask_vol + 1e-10)  # Sıfıra bölünmeyi önle
            
        return df

    def _add_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Makroekonomik göstergeler ekler"""
        # 1. Kripto makro verileri
        try:
            btc_dominance = self._get_btc_dominance()
            df['btc_dominance'] = btc_dominance
        except:
            df['btc_dominance'] = np.nan
            
        # 2. Global makro veriler (Örnek API)
        try:
            fed_rate = self._get_fed_rate()
            df['fed_rate'] = fed_rate
        except:
            df['fed_rate'] = np.nan
            
        # 3. Stablecoin verileri
        try:
            usdt_supply = self._get_usdt_supply()
            df['usdt_supply'] = usdt_supply
        except:
            df['usdt_supply'] = np.nan
            
        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """NaN değerleri ve aykırı değerleri temizler"""
        # NaN değerleri doldurma
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        
        # Aykırı değerleri kırpma
        for col in df.select_dtypes(include=[np.number]).columns:
            q1 = df[col].quantile(0.05)
            q3 = df[col].quantile(0.95)
            df[col] = np.clip(df[col], q1, q3)
            
        return df

    # Makro veri fonksiyonları (Örnek implementasyonlar)
    def _get_btc_dominance(self) -> float:
        url = "https://api.coingecko.com/api/v3/global"
        data = requests.get(url).json()
        return data['data']['market_cap_percentage']['btc']

    def _get_fed_rate(self) -> float:
        # Örnek API (Gerçekte FRED API kullanılabilir)
        url = "https://api.example.com/macro/fed-rate"
        return requests.get(url).json()['rate']

    def _get_usdt_supply(self) -> float:
        url = "https://api.example.com/stablecoins/usdt"
        return requests.get(url).json()['supply']

    def create_features_and_target(self, df: pd.DataFrame, future_bars: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
        """Özellik mühendisliği ve hedef değişken oluşturma"""
        # Hedef değişken (future_bars sonraki fiyat hareketi)
        df['future_close'] = df['close'].shift(-future_bars)
        df['target'] = np.where(
            df['future_close'] > df['close'] * 1.002, 1,  # 0.2%'den fazla artış
            np.where(
                df['future_close'] < df['close'] * 0.998, -1,  # 0.2%'den fazla düşüş
                0  # Nötr
            )
        )
        
        # Lag özellikleri
        for lag in [1, 3, 5]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            
        # Fiyat hareketi özellikleri
        df['price_change_5m'] = df['close'].pct_change(5)
        df['volume_change_5m'] = df['volume'].pct_change(5)
        
        # Temizleme
        df.dropna(inplace=True)
        features = df.drop(['target', 'future_close'], axis=1)
        target = df['target']
        
        return features, target