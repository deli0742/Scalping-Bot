import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import ccxt
from dataclasses import dataclass
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

@dataclass
class VolumeCluster:
    price_level: float
    volume: float
    type: str  # 'support' or 'resistance'

class VolumeProfileAnalyzer:
    def __init__(self, exchange: ccxt.Exchange):
        self.exchange = exchange
        self.historical_profiles = []
        
    def calculate_volume_profile(self, 
                               symbol: str, 
                               timeframe: str = '15m', 
                               lookback: int = 100,
                               num_clusters: int = 5) -> Dict:
        """
        Volume profile hesaplar ve önemli seviyeleri belirler
        
        Args:
            symbol: BTC/USDT gibi
            timeframe: 1m, 5m, 15m, etc.
            lookback: Analiz edilecek mum sayısı
            num_clusters: Tespit edilecek destek/direnç seviye sayısı
            
        Returns:
            {
                'vpoc': float,              # Volume Point of Control
                'value_area': (float, float), # 70% value area
                'clusters': List[VolumeCluster],
                'profile': Dict[float, float]  # Price -> Volume eşlemesi
            }
        """
        # OHLCV verisini al
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=lookback)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Volume profile hesapla
        profile = self._compute_raw_profile(df)
        vpoc = max(profile.items(), key=lambda x: x[1])[0]
        
        # Value Area (70%) hesapla
        value_area = self._calculate_value_area(profile)
        
        # Önemli seviyeleri kümele
        clusters = self._find_volume_clusters(profile, num_clusters)
        
        # Sonucu sakla
        result = {
            'vpoc': vpoc,
            'value_area': value_area,
            'clusters': clusters,
            'profile': profile,
            'timestamp': pd.Timestamp.now()
        }
        self.historical_profiles.append(result)
        
        return result
    
    def _compute_raw_profile(self, df: pd.DataFrame) -> Dict[float, float]:
        """Ham volume profile hesaplar"""
        profile = {}
        for _, row in df.iterrows():
            price_range = np.linspace(row['low'], row['high'], 100)
            volume_distribution = row['volume'] / len(price_range)
            
            for price in price_range:
                price = round(price, 2)  # Fiyat hassasiyeti
                if price in profile:
                    profile[price] += volume_distribution
                else:
                    profile[price] = volume_distribution
                    
        return profile
    
    def _calculate_value_area(self, profile: Dict[float, float], 
                            percentage: float = 0.7) -> Tuple[float, float]:
        """Value Area (VA) hesaplar"""
        sorted_prices = sorted(profile.items(), key=lambda x: x[1], reverse=True)
        total_volume = sum(profile.values())
        target_volume = total_volume * percentage
        
        cumulative_volume = 0
        va_prices = []
        
        for price, vol in sorted_prices:
            if cumulative_volume < target_volume:
                cumulative_volume += vol
                va_prices.append(price)
            else:
                break
                
        return (min(va_prices), max(va_prices))
    
    def _find_volume_clusters(self, 
                            profile: Dict[float, float], 
                            n_clusters: int = 5) -> List[VolumeCluster]:
        """Volume kümeleme ile destek/direnç seviyeleri bulur"""
        prices = np.array(list(profile.keys())).reshape(-1, 1)
        volumes = np.array(list(profile.values()))
        
        # K-means kümeleme
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(prices, sample_weight=volumes)
        
        clusters = []
        for i in range(n_clusters):
            cluster_prices = prices[kmeans.labels_ == i]
            cluster_volumes = volumes[kmeans.labels_ == i]
            
            if len(cluster_prices) == 0:
                continue
                
            # Küme merkezi ve toplam volume
            center = float(np.mean(cluster_prices))
            total_volume = float(np.sum(cluster_volumes))
            
            # Küme türünü belirle (price action'a göre)
            if center < prices.mean():
                cluster_type = 'support'
            else:
                cluster_type = 'resistance'
                
            clusters.append(
                VolumeCluster(
                    price_level=center,
                    volume=total_volume,
                    type=cluster_type
                )
            )
        
        # Volume'a göre sırala
        clusters.sort(key=lambda x: x.volume, reverse=True)
        return clusters
    
    def visualize_profile(self, 
                        profile_data: Dict, 
                        symbol: str) -> go.Figure:
        """Volume profile'ı interaktif görselleştirir"""
        prices = list(profile_data['profile'].keys())
        volumes = list(profile_data['profile'].values())
        
        fig = go.Figure()
        
        # Volume profile çubukları
        fig.add_trace(go.Bar(
            x=volumes,
            y=prices,
            orientation='h',
            name='Volume',
            marker_color='rgba(55, 128, 191, 0.6)'
        ))
        
        # VPOC çizgisi
        fig.add_shape(
            type="line",
            x0=0, x1=max(volumes)*1.1,
            y0=profile_data['vpoc'], y1=profile_data['vpoc'],
            line=dict(color="Red", width=2, dash="dot"),
            name="VPOC"
        )
        
        # Value Area
        fig.add_shape(
            type="rect",
            x0=0, x1=max(volumes)*1.1,
            y0=profile_data['value_area'][0], 
            y1=profile_data['value_area'][1],
            line=dict(color="Green", width=1),
            fillcolor="rgba(0, 128, 0, 0.1)",
            name="70% Value Area"
        )
        
        # Kümeleme seviyeleri
        for cluster in profile_data['clusters']:
            fig.add_shape(
                type="line",
                x0=0, x1=max(volumes)*1.1,
                y0=cluster.price_level, y1=cluster.price_level,
                line=dict(
                    color="Orange" if cluster.type == 'support' else "Purple",
                    width=1,
                    dash="dashdot"
                ),
                name=f"{cluster.type.capitalize()} ({cluster.volume:,.2f})"
            )
        
        fig.update_layout(
            title=f"{symbol} Volume Profile (VPOC: {profile_data['vpoc']})",
            yaxis_title="Price",
            xaxis_title="Volume",
            showlegend=True,
            height=800
        )
        
        return fig
    
    def get_historical_levels(self, 
                            window: int = 10) -> Dict[str, List[float]]:
        """Tarihsel önemli seviyeleri getirir"""
        if len(self.historical_profiles) < window:
            window = len(self.historical_profiles)
            
        recent_profiles = self.historical_profiles[-window:]
        
        supports = []
        resistances = []
        
        for profile in recent_profiles:
            for cluster in profile['clusters']:
                if cluster.type == 'support':
                    supports.append(cluster.price_level)
                else:
                    resistances.append(cluster.price_level)
                    
        return {
            'supports': supports,
            'resistances': resistances,
            'vpocs': [p['vpoc'] for p in recent_profiles]
        }