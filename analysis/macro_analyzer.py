import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import ccxt
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MacroEvent:
    name: str
    impact: str  # 'high', 'medium', 'low'
    actual_value: float
    expected_value: float
    date: datetime
    source: str

class MacroAnalyzer:
    def __init__(self, exchange: ccxt.Exchange = None):
        self.exchange = exchange if exchange else ccxt.binance()
        self.api_keys = {
            'fred': 'YOUR_FRED_API_KEY',
            'coinmetrics': 'YOUR_COINMETRICS_API_KEY',
            'alternative': 'YOUR_ALTERNATIVE_API_KEY'
        }
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def get_fed_rates(self) -> pd.DataFrame:
        """FED faiz oranlarını getirir"""
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id=FEDFUNDS&api_key={self.api_keys['fred']}&file_type=json"
        data = requests.get(url).json()['observations']
        
        df = pd.DataFrame(data)
        df['value'] = df['value'].astype(float)
        df['date'] = pd.to_datetime(df['date'])
        return df.set_index('date')

    def get_btc_dominance(self) -> pd.DataFrame:
        """BTC dominance verisini getirir"""
        url = "https://alternative.me/api/v2/crypto/dominance/"
        response = requests.get(url).json()
        data = response['data']['dominations']
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'], unit='s')
        df['value'] = df['value'].astype(float)
        return df.set_index('date')

    def get_stablecoin_flows(self) -> Dict[str, pd.DataFrame]:
        """Major stablecoin arz değişimleri"""
        coins = ['USDT', 'USDC', 'BUSD', 'DAI']
        result = {}
        
        for coin in coins:
            url = f"https://api.coinmetrics.io/v4/timeseries/asset-metrics?assets={coin}&metrics=SupplyCurr&api_key={self.api_keys['coinmetrics']}"
            data = requests.get(url).json()['data']
            
            df = pd.DataFrame([(x['time'], x['values'][0]) for x in data],
                             columns=['date', 'supply'])
            df['date'] = pd.to_datetime(df['date'])
            df['supply'] = df['supply'].astype(float)
            result[coin] = df.set_index('date')
            
        return result

    def get_crypto_fear_greed(self) -> pd.DataFrame:
        """Korku-Şeker Endeksi"""
        url = "https://api.alternative.me/fng/?limit=90"
        data = requests.get(url).json()['data']
        
        df = pd.DataFrame(data)
        df['value'] = df['value'].astype(int)
        df['date'] = pd.to_datetime(df['timestamp'], unit='s')
        return df[['date', 'value']].set_index('date')

    def get_upcoming_events(self, days_ahead: int = 7) -> List[MacroEvent]:
        """Yaklaşan önemli ekonomik etkinlikler"""
        url = f"https://www.econoday.com/byweek.asp?cust=json&days={days_ahead}"
        events = requests.get(url).json()
        
        macro_events = []
        for event in events:
            macro_events.append(MacroEvent(
                name=event['title'],
                impact=event['impact'],
                actual_value=float(event['actual']) if event['actual'] else None,
                expected_value=float(event['forecast']) if event['forecast'] else None,
                date=datetime.strptime(event['date'], '%Y-%m-%d'),
                source="Econoday"
            ))
            
        return macro_events

    def get_macro_score(self, symbol: str = 'BTC/USDT') -> Dict[str, float]:
        """Tüm makro faktörler için normalleştirilmiş skor"""
        # Verileri topla
        fed_rates = self.get_fed_rates().resample('D').last().ffill()
        dominance = self.get_btc_dominance().resample('D').last()
        fear_greed = self.get_crypto_fear_greed().resample('D').last()
        stablecoins = self.get_stablecoin_flows()
        
        # Son 30 günlük değişimleri hesapla
        metrics = {
            'fed_rate_change': fed_rates['value'].pct_change(30).iloc[-1],
            'dominance_change': dominance['value'].pct_change(30).iloc[-1],
            'fear_greed': fear_greed['value'].iloc[-1],
            'stablecoin_supply_change': sum(
                [df['supply'].pct_change(30).iloc[-1] 
                for df in stablecoins.values()
            ) / len(stablecoins)
        }
        
        # Normalleştirme (0-1 arası)
        normalized = {
            'fed_rate': self._normalize(metrics['fed_rate_change'], -0.1, 0.1),
            'dominance': self._normalize(metrics['dominance_change'], -0.3, 0.3),
            'sentiment': metrics['fear_greed'] / 100,
            'liquidity': self._normalize(metrics['stablecoin_supply_change'], -0.2, 0.5)
        }
        
        # Ağırlıklı skor
        weights = {
            'fed_rate': 0.3,
            'dominance': 0.2,
            'sentiment': 0.25,
            'liquidity': 0.25
        }
        
        total_score = sum(normalized[k] * weights[k] for k in normalized)
        
        return {
            'score': total_score,
            'components': normalized,
            'timestamp': datetime.now()
        }

    def _normalize(self, value: float, min_val: float, max_val: float) -> float:
        """Değeri 0-1 aralığına normalize eder"""
        return max(0, min(1, (value - min_val) / (max_val - min_val)))

    def plot_macro_trends(self, symbol: str = 'BTC/USDT'):
        """Makro göstergeleri görselleştirir"""
        fig, axes = plt.subplots(4, 1, figsize=(12, 16))
        
        # 1. FED Faiz Oranları
        fed_data = self.get_fed_rates().last('365D')
        axes[0].plot(fed_data.index, fed_data['value'], label='FED Funds Rate', color='blue')
        axes[0].set_title('FED Faiz Oranları (Son 1 Yıl)')
        axes[0].grid(True)
        
        # 2. BTC Dominance
        dominance_data = self.get_btc_dominance().last('90D')
        axes[1].plot(dominance_data.index, dominance_data['value'], label='BTC Dominance', color='orange')
        axes[1].set_title('BTC Dominance (Son 3 Ay)')
        axes[1].grid(True)
        
        # 3. Fear & Greed Index
        fg_data = self.get_crypto_fear_greed()
        axes[2].plot(fg_data.index, fg_data['value'], label='Fear & Greed', color='green')
        axes[2].axhline(25, color='red', linestyle='--')
        axes[2].axhline(75, color='green', linestyle='--')
        axes[2].set_title('Korku-Şeker Endeksi')
        axes[2].grid(True)
        
        # 4. Stablecoin Arzı
        stable_data = self.get_stablecoin_flows()
        for coin, df in stable_data.items():
            axes[3].plot(df.last('180D').index, df.last('180D')['supply'], label=coin)
        axes[3].set_title('Stablecoin Arzı (Son 6 Ay)')
        axes[3].legend()
        axes[3].grid(True)
        
        plt.tight_layout()
        return fig

    def generate_trading_signal(self, symbol: str) -> Dict:
        """Makro verilere göre ticaret sinyali üretir"""
        score = self.get_macro_score(symbol)
        price_data = self.exchange.fetch_ohlcv(symbol, '1d', limit=30)
        prices = [x[4] for x in price_data]
        
        # Trend analizi
        short_term = prices[-5:]
        long_term = prices[-20:]
        trend = 'up' if np.mean(short_term) > np.mean(long_term) else 'down'
        
        # Sinyal oluşturma
        if score['score'] > 0.7 and trend == 'up':
            signal = 'strong_buy'
        elif score['score'] > 0.6 and trend == 'up':
            signal = 'buy'
        elif score['score'] < 0.3 and trend == 'down':
            signal = 'strong_sell'
        elif score['score'] < 0.4 and trend == 'down':
            signal = 'sell'
        else:
            signal = 'neutral'
            
        return {
            'signal': signal,
            'score': score['score'],
            'trend': trend,
            'timestamp': datetime.now()
        }