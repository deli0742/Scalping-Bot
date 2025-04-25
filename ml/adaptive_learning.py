import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.base import ClassifierMixin
from sklearn.metrics import f1_score
from dataclasses import dataclass
from enum import Enum
import joblib
from ml.hyperparameter_opt import HyperparameterOptimizer
from core.data_processor import DataProcessor
import warnings
warnings.filterwarnings('ignore')

class MarketRegime(Enum):
    TRENDING_UP = 1
    TRENDING_DOWN = 2
    VOLATILE = 3
    SIDEWAYS = 4

@dataclass
class ModelSnapshot:
    model: ClassifierMixin
    performance: float
    market_conditions: Dict
    timestamp: pd.Timestamp

class AdaptiveLearningSystem:
    def __init__(self, initial_model, exchange: ccxt.Exchange):
        self.current_model = initial_model
        self.model_versions = []
        self.optimizer = HyperparameterOptimizer()
        self.data_processor = DataProcessor(exchange)
        self.performance_threshold = 0.65  # Model yenileme eşiği
        self.drift_detection_window = 100  # Drift kontrol penceresi
        
    def detect_market_regime(self, symbol: str) -> MarketRegime:
        """Piyasa rejimini tespit eder"""
        ohlcv = self.data_processor.exchange.fetch_ohlcv(symbol, '1h', limit=50)
        closes = np.array([x[4] for x in ohlcv])
        
        # Trend analizi
        returns = np.diff(closes) / closes[:-1]
        trend_strength = np.mean(returns)
        
        # Volatilite analizi
        volatility = np.std(returns)
        
        if trend_strength > 0.002 and volatility < 0.02:
            return MarketRegime.TRENDING_UP
        elif trend_strength < -0.002 and volatility < 0.02:
            return MarketRegime.TRENDING_DOWN
        elif volatility > 0.03:
            return MarketRegime.VOLATILE
        else:
            return MarketRegime.SIDEWAYS

    def evaluate_model_performance(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Modelin güncel performansını ölçer"""
        preds = self.current_model.predict(X)
        return f1_score(y, preds, average='weighted')

    def detect_concept_drift(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Konsept drift kontrolü"""
        recent_perf = self.evaluate_model_performance(X[-self.drift_detection_window:], 
                                                    y[-self.drift_detection_window:])
        historical_perf = self.evaluate_model_performance(X[:-self.drift_detection_window], 
                                                        y[:-self.drift_detection_window])
        
        return (historical_perf - recent_perf) > 0.15  %15'ten fazla düşüş

    def adaptive_retrain(self, X: pd.DataFrame, y: pd.Series, symbol: str) -> Optional[ClassifierMixin]:
        """Akıllı yeniden eğitim mekanizması"""
        current_perf = self.evaluate_model_performance(X, y)
        
        # Performans eşiği kontrolü
        if current_perf >= self.performance_threshold:
            return None
            
        # Piyasa rejimini belirle
        market_regime = self.detect_market_regime(symbol)
        print(f"Piyasa rejimi değişti: {market_regime.name}")
        
        # Yeni veri özellikleri ekle
        X_enhanced = self._enhance_features(X, market_regime)
        
        # Mevcut modelin hiperparametre optimizasyonu
        best_params, best_score = self.optimizer.adaptive_reoptimization(
            X_enhanced, y,
            previous_params=self.current_model.get_params()
        )
        
        # Modeli güncelle
        self.current_model.set_params(**best_params)
        self.current_model.fit(X_enhanced, y)
        
        # Snapshot kaydet
        self._save_model_snapshot(X_enhanced, y, market_regime)
        
        return self.current_model

    def _enhance_features(self, X: pd.DataFrame, regime: MarketRegime) -> pd.DataFrame:
        """Piyasa rejimine özel özellikler ekler"""
        X = X.copy()
        
        if regime == MarketRegime.TRENDING_UP:
            X['trend_strength'] = X['close'].pct_change(5).rolling(10).mean()
        elif regime == MarketRegime.TRENDING_DOWN:
            X['trend_strength'] = X['close'].pct_change(5).rolling(10).mean()
        elif regime == MarketRegime.VOLATILE:
            X['volatility_index'] = X['high'] / X['low'] - 1
        else:  # SIDEWAYS
            X['mean_reversion'] = (X['close'] - X['close'].rolling(20).mean()) / X['close'].rolling(20).std()
            
        return X.fillna(0)

    def _save_model_snapshot(self, X: pd.DataFrame, y: pd.Series, regime: MarketRegime):
        """Model anlık görüntüsünü kaydeder"""
        snapshot = ModelSnapshot(
            model=joblib.dumps(self.current_model),
            performance=self.evaluate_model_performance(X, y),
            market_conditions={
                'regime': regime.name,
                'volatility': X['volatility'].mean(),
                'volume': X['volume'].mean()
            },
            timestamp=pd.Timestamp.now()
        )
        self.model_versions.append(snapshot)
        
        # En fazla 10 snapshot sakla
        if len(self.model_versions) > 10:
            self.model_versions.pop(0)

    def rollback_model(self, n_versions: int = 1) -> bool:
        """Önceki bir model versiyonuna döner"""
        if len(self.model_versions) >= n_versions:
            snapshot = self.model_versions[-n_versions]
            self.current_model = joblib.loads(snapshot.model)
            return True
        return False

    def get_current_model_info(self) -> Dict:
        """Güncel model bilgilerini verir"""
        if not self.model_versions:
            return {}
            
        latest = self.model_versions[-1]
        return {
            'performance': latest.performance,
            'market_conditions': latest.market_conditions,
            'timestamp': str(latest.timestamp)
        }