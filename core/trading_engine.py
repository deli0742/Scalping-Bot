import ccxt
import numpy as np
from tensorflow.keras.models import load_model
from analysis.orderbook_analyzer import OrderbookAnalyzer
from analysis.macro_analyzer import MacroAnalyzer

class TradingEngine:
    def __init__(self, api_key, secret):
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': secret,
            'options': {'defaultType': 'future'}
        })
        self.model = load_model('adaptive_model.h5')
        self.orderbook_analyzer = OrderbookAnalyzer()
        self.macro_analyzer = MacroAnalyzer()

    async def analyze_market(self, symbol):
        # Order book derinlik analizi
        orderbook = await self.exchange.fetch_order_book(symbol)
        liquidity_ratio = self.orderbook_analyzer.calculate_liquidity(orderbook)
        
        # Makro veriler
        macro_score = self.macro_analyzer.get_current_score()
        
        # Volume profile
        ohlcv = self.exchange.fetch_ohlcv(symbol, '15m', limit=100)
        volume_data = [x[5] for x in ohlcv]
        
        return {
            'liquidity': liquidity_ratio,
            'macro': macro_score,
            'volume_profile': self.calculate_volume_profile(volume_data)
        }

    def execute_trade(self, signal, symbol, amount):
        # Akıllı order execution
        if signal > 0.7:  # Güçlü al sinyali
            self.exchange.create_order(
                symbol, 'limit', 'buy', amount, 
                price=self.calculate_optimal_entry(symbol)
            )
        elif signal < 0.3:  # Güçlü sat sinyali
            # Hedge stratejisi
            self.place_oco_order(symbol, amount)