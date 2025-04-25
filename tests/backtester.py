import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import quantstats as qs
import backtrader as bt
from backtrader.feeds import PandasData
from core.data_processor import DataProcessor
from analysis.volume_profile import VolumeProfileAnalyzer
from analysis.macro_analyzer import MacroAnalyzer
import matplotlib.pyplot as plt
import json
import os
from multiprocessing import Pool
import warnings
warnings.filterwarnings('ignore')

class BacktestEngine:
    def __init__(self, exchange: ccxt.Exchange, strategy_class: bt.Strategy):
        self.exchange = exchange
        self.strategy_class = strategy_class
        self.data_processor = DataProcessor(exchange)
        self.vp_analyzer = VolumeProfileAnalyzer(exchange)
        self.macro_analyzer = MacroAnalyzer(exchange)
        self.results = {}

    def fetch_historical_data(self, symbol: str, timeframe: str, 
                            start_date: str, end_date: str) -> pd.DataFrame:
        """Binance'tan tarihsel veri çeker"""
        since = self.exchange.parse8601(start_date)
        end = self.exchange.parse8601(end_date)
        all_ohlcv = []
        
        while since < end:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol, 
                timeframe, 
                since=since,
                limit=1000
            )
            if not ohlcv:
                break
            since = ohlcv[-1][0] + 1
            all_ohlcv += ohlcv
        
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df.loc[start_date:end_date]

    def enhance_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Veriyi teknik göstergeler ve volume profile ile zenginleştirir"""
        # Teknik göstergeler
        df = self.data_processor._add_technical_indicators(df)
        
        # Volume profile
        vp_data = self.vp_analyzer.calculate_volume_profile(symbol, df=df)
        df['vpoc'] = vp_data['vpoc']
        df['va_low'] = vp_data['value_area'][0]
        df['va_high'] = vp_data['value_area'][1]
        
        # Makro veriler (simüle edilmiş)
        df['fed_rate'] = self.macro_analyzer.get_fed_rates().reindex(df.index, method='ffill')
        df['btc_dominance'] = self.macro_analyzer.get_btc_dominance().reindex(df.index, method='ffill')
        
        return df.dropna()

    def prepare_backtest_data(self, symbol: str, timeframe: str,
                            start_date: str, end_date: str) -> pd.DataFrame:
        """Backtest için veri hazırlar"""
        raw_data = self.fetch_historical_data(symbol, timeframe, start_date, end_date)
        enhanced_data = self.enhance_data(raw_data, symbol)
        return enhanced_data

    def run_backtest(self, symbol: str, timeframe: str,
                    start_date: str, end_date: str,
                    strategy_params: Dict = None,
                    commission: float = 0.0004,  # Binance komisyonu
                    stake: float = 1000,  # USDT cinsinden
                    plot: bool = False) -> Dict:
        """Backtrader ile backtest çalıştırır"""
        # Veriyi hazırla
        data_df = self.prepare_backtest_data(symbol, timeframe, start_date, end_date)
        
        # Backtrader için veri formatını ayarla
        class BinanceData(PandasData):
            lines = ('vpoc', 'va_low', 'va_high', 'fed_rate', 'btc_dominance')
            params = (
                ('datetime', None),
                ('open', 'open'),
                ('high', 'high'),
                ('low', 'low'),
                ('close', 'close'),
                ('volume', 'volume'),
                ('vpoc', 'vpoc'),
                ('va_low', 'va_low'),
                ('va_high', 'va_high'),
                ('fed_rate', 'fed_rate'),
                ('btc_dominance', 'btc_dominance'),
                ('timeframe', bt.TimeFrame.Minutes),
                ('compression', int(timeframe[:-1]))
            )
        
        data = BinanceData(dataname=data_df)
        
        # Cerebro motorunu başlat
        cerebro = bt.Cerebro(stdstats=False, optreturn=False)
        cerebro.adddata(data)
        cerebro.broker.setcash(stake)
        cerebro.broker.setcommission(commission=commission)
        
        # Strateji ekle
        if strategy_params:
            cerebro.addstrategy(self.strategy_class, **strategy_params)
        else:
            cerebro.addstrategy(self.strategy_class)
        
        # Analyzer'lar ekle
        cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.Transactions, _name='transactions')
        
        # Backtest'i çalıştır
        result = cerebro.run()
        
        # Sonuçları topla
        strat = result[0]
        pyfoliozer = strat.analyzers.getbyname('pyfolio')
        returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
        
        # QuantStats raporu oluştur
        qs.reports.full(returns, output='backtest_result.html')
        
        # Sonuçları kaydet
        self.results = {
            'performance': {
                'return': cerebro.broker.getvalue() / stake - 1,
                'sharpe': strat.analyzers.sharpe.get_analysis()['sharperatio'],
                'max_drawdown': strat.analyzers.drawdown.get_analysis()['max']['drawdown'],
                'win_rate': self._calculate_win_rate(strat.analyzers.trades.get_analysis())
            },
            'transactions': self._process_transactions(strat.analyzers.transactions.get_analysis()),
            'report_path': os.path.abspath('backtest_result.html')
        }
        
        # Grafik oluştur (isteğe bağlı)
        if plot:
            cerebro.plot(style='candlestick', volume=False)
        
        return self.results

    def _calculate_win_rate(self, trade_analysis: Dict) -> float:
        """Kazanç oranını hesaplar"""
        if not trade_analysis or 'won' not in trade_analysis:
            return 0.0
        total = trade_analysis['won']['total'] + trade_analysis['lost']['total']
        return trade_analysis['won']['total'] / total if total > 0 else 0.0

    def _process_transactions(self, transactions: Dict) -> List[Dict]:
        """İşlemleri işlenebilir formata çevirir"""
        processed = []
        for tx in transactions:
            processed.append({
                'date': tx[0].datetime(),
                'size': tx[0].size,
                'price': tx[0].price,
                'value': tx[0].value,
                'commission': tx[0].comm,
                'pnl': tx[0].pnl
            })
        return processed

    def optimize_strategy(self, symbol: str, timeframe: str,
                        start_date: str, end_date: str,
                        params_space: Dict,
                        max_workers: int = 4) -> Dict:
        """Strateji parametrelerini optimize eder"""
        data_df = self.prepare_backtest_data(symbol, timeframe, start_date, end_date)
        
        def optimize_worker(params):
            cerebro = bt.Cerebro(stdstats=False, optreturn=False)
            data = PandasData(dataname=data_df)
            cerebro.adddata(data)
            cerebro.broker.setcash(1000)
            cerebro.broker.setcommission(commission=0.0004)
            cerebro.addstrategy(self.strategy_class, **params)
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            result = cerebro.run()[0]
            sharpe = result.analyzers.sharpe.get_analysis()['sharperatio']
            return {**params, 'sharpe': sharpe}
        
        # Parametre kombinasyonlarını oluştur
        param_combinations = self._generate_param_combinations(params_space)
        
        # Paralel optimizasyon
        with Pool(max_workers) as pool:
            results = pool.map(optimize_worker, param_combinations)
        
        # En iyi sonucu bul
        best_result = max(results, key=lambda x: x['sharpe'])
        return best_result

    def _generate_param_combinations(self, params_space: Dict) -> List[Dict]:
        """Parametre uzayından kombinasyonlar üretir"""
        from itertools import product
        
        keys = params_space.keys()
        values = params_space.values()
        combinations = product(*values)
        
        return [dict(zip(keys, combo)) for combo in combinations]

    def save_results(self, filepath: str = 'backtest_results.json'):
        """Sonuçları JSON olarak kaydeder"""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)

    def generate_report(self):
        """HTML raporu oluşturur ve tarayıcıda açar"""
        import webbrowser
        webbrowser.open(self.results['report_path'])