import numpy as np
import pandas as pd
import ccxt
from typing import Dict, Tuple, Optional
from enum import Enum

class RiskLevel(Enum):
    LOW = 1
    MODERATE = 2
    HIGH = 3

class RiskManager:
    def __init__(self, exchange: ccxt.Exchange, capital: float, risk_level: RiskLevel = RiskLevel.MODERATE):
        self.exchange = exchange
        self.capital = capital
        self.risk_level = risk_level
        self.position_sizes = {}
        self.max_drawdown = 0.1  # %10 maksimum çöküş
        
        # Risk parametreleri
        self._set_risk_parameters()

    def _set_risk_parameters(self):
        """Risk seviyesine göre parametreleri ayarlar"""
        if self.risk_level == RiskLevel.LOW:
            self.max_position_size = 0.05  # %5
            self.leverage = 3
            self.stop_loss_pct = 0.005  # %0.5
            self.take_profit_pct = 0.015  # %1.5
        elif self.risk_level == RiskLevel.MODERATE:
            self.max_position_size = 0.1  # %10
            self.leverage = 5
            self.stop_loss_pct = 0.01  # %1
            self.take_profit_pct = 0.03  # %3
        else:  # HIGH
            self.max_position_size = 0.2  # %20
            self.leverage = 10
            self.stop_loss_pct = 0.02  # %2
            self.take_profit_pct = 0.05  # %5

    def calculate_position_size(self, symbol: str, entry_price: float) -> Dict[str, float]:
        """Pozisyon büyüklüğünü ve risk parametrelerini hesaplar"""
        # Piyasa koşullarını analiz et
        volatility = self._get_volatility(symbol)
        liquidity = self._get_liquidity(symbol)
        
        # Dinamik risk ayarlaması
        adjusted_risk = self._adjust_for_market_conditions(volatility, liquidity)
        
        # Pozisyon boyutu hesaplama
        max_risk_amount = self.capital * self.max_position_size * adjusted_risk
        contract_size = self._get_contract_size(symbol)
        
        # Futures için sözleşme miktarı
        position_size = (max_risk_amount / entry_price) * self.leverage
        position_size = min(position_size, liquidity * 0.1)  # Likiditenin %10'unu geçmemek
        
        # Kontrat boyutuna göre ayarla
        position_size = round(position_size / contract_size) * contract_size
        
        self.position_sizes[symbol] = {
            'size': position_size,
            'leverage': self.leverage,
            'stop_loss': entry_price * (1 - self.stop_loss_pct),
            'take_profit': entry_price * (1 + self.take_profit_pct),
            'risk_score': adjusted_risk
        }
        
        return self.position_sizes[symbol]

    def _adjust_for_market_conditions(self, volatility: float, liquidity: float) -> float:
        """Piyasa koşullarına göre riski ayarlar"""
        # Volatilite yüksekse riski azalt
        volatility_factor = 1 / (1 + np.log(1 + volatility))
        
        # Likidite düşükse riski azalt
        liquidity_factor = min(1, liquidity / 1_000_000)  # 1M USD likidite için normalize
        
        return volatility_factor * liquidity_factor

    def _get_volatility(self, symbol: str, lookback: int = 20) -> float:
        """Son 20 periyodun volatilitesini hesaplar"""
        ohlcv = self.exchange.fetch_ohlcv(symbol, '1d', limit=lookback)
        closes = np.array([x[4] for x in ohlcv])
        returns = np.diff(closes) / closes[:-1]
        return np.std(returns) * np.sqrt(365)  # Yıllık volatilite

    def _get_liquidity(self, symbol: str) -> float:
        """Order book derinliğine göre likiditeyi ölçer"""
        orderbook = self.exchange.fetch_order_book(symbol)
        return sum([x[1] for x in orderbook['bids'][:10]])  # İlk 10 bid'in toplamı

    def _get_contract_size(self, symbol: str) -> float:
        """Futures kontrat boyutunu getirir"""
        markets = self.exchange.load_markets()
        return markets[symbol]['contractSize']

    def generate_orders(self, symbol: str, signal: float) -> Optional[Dict]:
        """Risk parametrelerine uygun emirler oluşturur"""
        if symbol not in self.position_sizes:
            return None

        position = self.position_sizes[symbol]
        current_price = self.exchange.fetch_ticker(symbol)['last']
        
        # Sinyal gücüne göre pozisyon ayarlaması
        size_multiplier = min(1.0, abs(signal) * 2)  # Sinyal 0-0.5 aralığında normalize
        
        # Emir yapısı
        orders = {
            'main_order': {
                'symbol': symbol,
                'type': 'limit',
                'side': 'buy' if signal > 0 else 'sell',
                'amount': position['size'] * size_multiplier,
                'price': current_price,
                'leverage': position['leverage'],
                'reduceOnly': False
            },
            'stop_loss': {
                'symbol': symbol,
                'type': 'stop_market',
                'side': 'sell' if signal > 0 else 'buy',
                'amount': position['size'] * size_multiplier,
                'stopPrice': position['stop_loss'],
                'reduceOnly': True
            },
            'take_profit': {
                'symbol': symbol,
                'type': 'take_profit_market',
                'side': 'sell' if signal > 0 else 'buy',
                'amount': position['size'] * size_multiplier,
                'stopPrice': position['take_profit'],
                'reduceOnly': True
            }
        }
        
        return orders

    def check_margin_level(self) -> bool:
        """Margin seviyesini kontrol eder"""
        balance = self.exchange.fetch_balance()
        margin_level = balance['info']['marginLevel']
        return float(margin_level) > 0.5  # %50 margin seviyesi

    def adjust_leverage(self, symbol: str, new_leverage: int):
        """Pozisyon kaldıracını ayarlar"""
        self.exchange.set_leverage(new_leverage, symbol)
        self.leverage = new_leverage

    def monitor_drawdown(self, current_equity: float) -> bool:
        """Çöküşü izler ve aşarsa tüm pozisyonları kapatır"""
        drawdown = (self.capital - current_equity) / self.capital
        if drawdown > self.max_drawdown:
            self.close_all_positions()
            return False
        return True

    def close_all_positions(self):
        """Tüm pozisyonları güvenli şekilde kapatır"""
        positions = self.exchange.fetch_positions()
        for pos in positions:
            if float(pos['contracts']) > 0:
                self.exchange.create_order(
                    pos['symbol'],
                    'market',
                    'sell' if pos['side'] == 'long' else 'buy',
                    abs(float(pos['contracts'])),
                    params={'reduceOnly': True}
                )