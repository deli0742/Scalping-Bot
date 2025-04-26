import os
import requests
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class MacroEvent:
    name: str
    impact: str  # 'high', 'medium', 'low'
    actual_value: Optional[float]
    expected_value: Optional[float]
    date: datetime
    source: str

class MacroAnalyzer:
    def __init__(self, exchange=None):
        self.exchange = exchange
        self.api_keys = {
            'fred': os.getenv('FRED_API_KEY', ''),
            'coinmetrics': os.getenv('COINMETRICS_API_KEY', ''),
            'alternative': os.getenv('ALTERNATIVE_API_KEY', '')
        }

    def get_fed_rates(self) -> pd.DataFrame:
        """Fetch FED funds rate observations; return zero series on error."""
        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id=FEDFUNDS&api_key={self.api_keys['fred']}&file_type=json"
        )
        try:
            resp = requests.get(url)
            data = resp.json()
            obs = data.get('observations')
            if not obs:
                raise ValueError("No observations returned from FED API")
            df = pd.DataFrame(obs)
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            return df.set_index('date').sort_index()
        except Exception:
            # return 30-day zero series
            dates = pd.date_range(end=datetime.now(), periods=30)
            return pd.DataFrame({'value': [0]*len(dates)}, index=dates)

    def normalize_change(self, series: pd.Series, window: int, min_val: float, max_val: float) -> float:
        """Compute pct change over window and normalize to [0,1]."""
        try:
            change = series.pct_change(periods=window).iloc[-1]
            return max(0.0, min(1.0, (change - min_val) / (max_val - min_val)))
        except Exception:
            return 0.5

    def get_macro_score(self, symbol: str = 'BTC/USDT') -> Dict[str, float]:
        """Combine multiple macro factors into a weighted score."""
        # FED rate change
        fed_series = self.get_fed_rates()['value'].resample('D').last().ffill()
        fed_score = self.normalize_change(fed_series, window=30, min_val=-0.1, max_val=0.1)

        # Placeholder for other factors (later replace with real API calls)
        dominance_score = 0.5
        sentiment_score = 0.5
        liquidity_score = 0.5

        weights = {'fed_rate': 0.3, 'dominance': 0.2, 'sentiment': 0.25, 'liquidity': 0.25}
        total = (
            fed_score * weights['fed_rate'] +
            dominance_score * weights['dominance'] +
            sentiment_score * weights['sentiment'] +
            liquidity_score * weights['liquidity']
        )

        return {
            'score': total,
            'components': {
                'fed_rate': fed_score,
                'dominance': dominance_score,
                'sentiment': sentiment_score,
                'liquidity': liquidity_score
            }
        }

    def get_upcoming_events(self, days_ahead: int = 7) -> List[MacroEvent]:
        """Fetch upcoming economic events."""
        # Implementation omitted for brevity; can return empty list or real data
        return []

    # Additional methods (e.g., historical plotting) can be implemented as needed
