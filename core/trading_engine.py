import ccxt
import joblib
import numpy as np
import time
import warnings
# Suppress sklearn feature names warning
warnings.filterwarnings('ignore', message='X does not have valid feature names')
from analysis.orderbook_analyzer import OrderbookAnalyzer
from analysis.macro_analyzer import MacroAnalyzer

class TradingEngine:
    def __init__(self, api_key, secret, model_path='adaptive_model.pkl'):
        # Binance futures connection
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': secret,
            'options': {'defaultType': 'future'}
        })

        # Load trained model
        data = joblib.load(model_path)
        self.model = data['model']
        self.best_params = data.get('best_params', {})
        print(f"[✓] Model loaded: {model_path}")

        # Analysis components
        self.orderbook_analyzer = OrderbookAnalyzer()
        self.macro_analyzer     = MacroAnalyzer()

        # Control flags
        self._running = False
        self._symbol = None

    def analyze_market(self, symbol: str) -> dict:
        """Analyze orderbook, macro score, and volume profile."""
        ob = self.exchange.fetch_order_book(symbol)
        liquidity = self.orderbook_analyzer.calculate_liquidity(ob)
        macro_score = self.macro_analyzer.get_macro_score(symbol)['score']
        ohlcv = self.exchange.fetch_ohlcv(symbol, '15m', limit=100)
        volumes = np.array([x[5] for x in ohlcv])
        vp = self.calculate_volume_profile(volumes)
        return {'liquidity': liquidity, 'macro': macro_score, 'volume_profile': vp}

    def generate_signal(self, symbol: str) -> dict:
        """Generate trading signal based on analyzed features."""
        features = self.analyze_market(symbol)
        # Liquidity
        raw_liq = features.get('liquidity', 0.0)
        if isinstance(raw_liq, dict):
            vals = [v for v in raw_liq.values() if isinstance(v, (int, float))]
            liquidity = float(sum(vals) / len(vals)) if vals else 0.0
        else:
            try:
                liquidity = float(raw_liq)
            except:
                liquidity = 0.0
        # Macro
        try:
            macro = float(features.get('macro', 0.0))
        except:
            macro = 0.0
        # Volume profile
        vp = features.get('volume_profile', {}) or {}
        try:
            p20 = float(vp.get('p20', 0.0))
            p40 = float(vp.get('p40', 0.0))
            p60 = float(vp.get('p60', 0.0))
            p80 = float(vp.get('p80', 0.0))
        except:
            p20 = p40 = p60 = p80 = 0.0
        # Build feature vector
        feats = [liquidity, macro, p20, p40, p60, p80]
        expected = getattr(self.model, 'n_features_in_', len(feats))
        if len(feats) < expected:
            feats += [0.0] * (expected - len(feats))
        else:
            feats = feats[:expected]
        arr = np.array([feats], dtype=float)
        probs = self.model.predict_proba(arr)[0]
        return {'signal': float(probs[1]), 'probabilities': probs.tolist()}

    def execute_trade(self, signal: float, symbol: str, amount: float):
        """Execute order based on signal thresholds."""
        if signal > 0.7:
            price = self.calculate_optimal_entry(symbol)
            return self.exchange.create_order(symbol, 'limit', 'buy', amount, price)
        elif signal < 0.3:
            return self.place_oco_order(symbol, amount)
        return None

    def calculate_volume_profile(self, volumes: np.ndarray) -> dict:
        """Compute volume percentiles."""
        try:
            p = np.percentile(volumes, [20, 40, 60, 80])
            return {'p20': float(p[0]), 'p40': float(p[1]), 'p60': float(p[2]), 'p80': float(p[3])}
        except:
            return {'p20': 0.0, 'p40': 0.0, 'p60': 0.0, 'p80': 0.0}

    def calculate_optimal_entry(self, symbol: str) -> float:
        """Return last price as entry point."""
        return float(self.exchange.fetch_ticker(symbol)['last'])

    def place_oco_order(self, symbol: str, amount: float):
        """Place a sample OCO sell order."""
        ticker = self.exchange.fetch_ticker(symbol)
        price = float(ticker['last'])
        return self.exchange.create_oco_order(symbol, 'sell', amount, price * 1.02, price * 0.98)

    def run(self, symbol: str, risk: float = 1.0, strategy: str = 'hybrid', interval: int = 60):
        """Start the bot loop with error-handled trade execution."""
        self._running = True
        self._symbol = symbol
        print(f"Bot started: {symbol}, risk={risk}, strategy={strategy}")
        while self._running:
            try:
                sig = self.generate_signal(symbol)
                print(f"[{symbol}] Signal: {sig['signal']:.2f}")
                try:
                    order = self.execute_trade(sig['signal'], symbol, amount=risk)
                    print(f"Order result: {order}")
                except ccxt.InsufficientFunds as ie:
                    print(f"⚠️ Insufficient margin: {ie}")
                except Exception as oe:
                    print(f"Trade execution error: {oe}")
            except Exception as e:
                print(f"Error in bot loop: {e}")
            time.sleep(interval)
        print("Bot stopped.")

    def stop(self):
        """Stop the bot loop."""
        self._running = False

    def get_current_signals(self, symbol: str = None) -> dict:
        """Fetch current metrics and signal."""
        s = symbol or self._symbol or 'BTC/USDT'
        tk = self.exchange.fetch_ticker(s)
        sig = self.generate_signal(s)
        return {
            'current_price': float(tk['last']),
            'price_change': float(tk.get('percentage', 0)),
            'signal': sig['signal'],
            'probabilities': sig['probabilities'],
            'recent_trades': []
        }

    def get_macro_analysis(self, symbol: str = None) -> dict:
        """Fetch macro score analysis."""
        s = symbol or self._symbol or 'BTC/USDT'
        return self.macro_analyzer.get_macro_score(s)

    def get_orderbook_analysis(self, symbol: str = None) -> dict:
        """Fetch orderbook liquidity analysis."""
        s = symbol or self._symbol or 'BTC/USDT'
        ob = self.exchange.fetch_order_book(s)
        return {'liquidity': self.orderbook_analyzer.calculate_liquidity(ob)}
