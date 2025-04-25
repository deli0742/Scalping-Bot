import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import quantstats as qs
from scipy import stats
from sklearn.metrics import confusion_matrix
from core.data_processor import DataProcessor
from tests.backtester import BacktestEngine
from tests.reporter import ReportGenerator
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PerformanceTester:
    def __init__(self, exchange, strategy_class):
        self.exchange = exchange
        self.strategy_class = strategy_class
        self.backtester = BacktestEngine(exchange, strategy_class)
        self.reporter = ReportGenerator()
        self.data_processor = DataProcessor(exchange)

    def run_comprehensive_test(self, symbol: str, timeframe: str,
                             start_date: str, end_date: str,
                             benchmark: str = 'BTC/USDT') -> Dict:
        """
        Kapsamlı performans testi yürütür:
        1. Temel backtest
        2. Walk-forward analizi
        3. Monte Carlo simülasyonu
        4. Benchmark karşılaştırması
        """
        results = {}
        
        # 1. Temel backtest
        print("Running base backtest...")
        base_results = self.backtester.run_backtest(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        results['base'] = base_results
        
        # 2. Walk-forward analizi
        print("Running walk-forward analysis...")
        wfa_results = self.walk_forward_analysis(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        results['walk_forward'] = wfa_results
        
        # 3. Monte Carlo simülasyonu
        print("Running Monte Carlo simulation...")
        mc_results = self.monte_carlo_simulation(
            returns=base_results['returns'],
            num_simulations=1000
        )
        results['monte_carlo'] = mc_results
        
        # 4. Benchmark karşılaştırması
        print("Running benchmark comparison...")
        benchmark_results = self.benchmark_comparison(
            strategy_returns=base_results['returns'],
            symbol=symbol,
            benchmark=benchmark,
            start_date=start_date,
            end_date=end_date
        )
        results['benchmark'] = benchmark_results
        
        # Rapor oluştur
        print("Generating reports...")
        report_path = self.reporter.generate_all_reports(results, self.strategy_class.__name__)
        
        return {
            'results': results,
            'report_path': report_path
        }

    def walk_forward_analysis(self, symbol: str, timeframe: str,
                             start_date: str, end_date: str,
                             train_ratio: float = 0.7) -> Dict:
        """
        Walk-forward optimizasyonu yapar:
        1. Veriyi train/test olarak böler
        2. Train sette optimizasyon
        3. Test sette out-of-sample test
        """
        # Tüm veriyi yükle
        full_data = self.backtester.prepare_backtest_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        # Train/test bölme
        split_idx = int(len(full_data) * train_ratio)
        train_data = full_data.iloc[:split_idx]
        test_data = full_data.iloc[split_idx:]
        
        # Parametre optimizasyonu (train sette)
        print("Optimizing parameters on training set...")
        optimal_params = self.backtester.optimize_strategy(
            data=train_data,
            params_space=self._get_strategy_params_space()
        )
        
        # Test sette out-of-sample backtest
        print("Running out-of-sample test...")
        test_results = self.backtester.run_backtest(
            data=test_data,
            strategy_params=optimal_params
        )
        
        return {
            'optimal_params': optimal_params,
            'test_results': test_results,
            'train_period': (train_data.index[0], train_data.index[-1]),
            'test_period': (test_data.index[0], test_data.index[-1])
        }

    def monte_carlo_simulation(self, returns: pd.Series,
                              num_simulations: int = 1000,
                              confidence_level: float = 0.95) -> Dict:
        """
        Monte Carlo simülasyonu ile risk analizi yapar
        """
        # Bootstrap yöntemiyle simülasyon
        simulated_returns = []
        for _ in range(num_simulations):
            # Rastgele örnekleme (with replacement)
            sampled_returns = np.random.choice(
                returns, 
                size=len(returns),
                replace=True
            )
            cumulative_return = np.prod(1 + sampled_returns) - 1
            simulated_returns.append(cumulative_return)
        
        # Risk metrikleri hesapla
        var = np.percentile(simulated_returns, 100 * (1 - confidence_level))
        cvar = np.mean([x for x in simulated_returns if x <= var])
        
        return {
            'expected_return': np.mean(simulated_returns),
            'value_at_risk': var,
            'conditional_var': cvar,
            'simulated_returns': simulated_returns,
            'confidence_level': confidence_level
        }

    def benchmark_comparison(self, strategy_returns: pd.Series,
                            symbol: str, benchmark: str,
                            start_date: str, end_date: str) -> Dict:
        """
        Stratejiyi benchmark ile karşılaştırır
        """
        # Benchmark verisini al
        benchmark_data = self.backtester.fetch_historical_data(
            symbol=benchmark,
            timeframe='1d',
            start_date=start_date,
            end_date=end_date
        )
        benchmark_returns = benchmark_data['close'].pct_change().dropna()
        
        # Strateji returns'ünü benchmark ile aynı indekse getir
        aligned_strategy = strategy_returns.reindex(benchmark_returns.index, method='ffill')
        
        # Performans karşılaştırması
        comparison = qs.reports.metrics(
            returns=aligned_strategy,
            benchmark=benchmark_returns,
            display=False
        )
        
        # İstatistiksel testler
        _, pvalue = stats.ttest_ind(
            aligned_strategy.dropna(),
            benchmark_returns.dropna(),
            equal_var=False
        )
        
        return {
            'benchmark': benchmark,
            'comparison_metrics': comparison.to_dict(),
            't_test_pvalue': pvalue,
            'strategy_returns': aligned_strategy,
            'benchmark_returns': benchmark_returns
        }

    def sensitivity_analysis(self, symbol: str, timeframe: str,
                            start_date: str, end_date: str,
                            param_ranges: Dict) -> Dict:
        """
        Parametre duyarlılık analizi yapar
        """
        results = []
        
        # Parametre kombinasyonlarını oluştur
        from itertools import product
        param_names = param_ranges.keys()
        param_values = param_ranges.values()
        
        for combination in product(*param_values):
            params = dict(zip(param_names, combination))
            
            # Backtest çalıştır
            test_results = self.backtester.run_backtest(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                strategy_params=params
            )
            
            # Sonuçları kaydet
            results.append({
                'params': params,
                'sharpe': test_results['performance']['sharpe'],
                'return': test_results['performance']['return'],
                'max_drawdown': test_results['performance']['max_drawdown']
            })
        
        return pd.DataFrame(results)

    def _get_strategy_params_space(self) -> Dict:
        """Strateji parametre uzayını tanımlar"""
        return {
            'rsi_period': [14, 21, 28],
            'atr_multiplier': [1.5, 2.0, 2.5],
            'take_profit': [0.005, 0.01, 0.015],
            'stop_loss': [0.005, 0.01, 0.015]
        }

    def plot_sensitivity_results(self, sensitivity_results: pd.DataFrame):
        """Parametre duyarlılık sonuçlarını görselleştirir"""
        param_names = [col for col in sensitivity_results.columns if col != 'results']
        
        plt.figure(figsize=(15, 10))
        for i, metric in enumerate(['sharpe', 'return', 'max_drawdown'], 1):
            plt.subplot(3, 1, i)
            
            for param in param_names:
                sns.lineplot(
                    data=sensitivity_results,
                    x=param,
                    y=metric,
                    label=param
                )
            
            plt.title(f'Parameter Sensitivity - {metric.upper()}')
            plt.legend()
        
        plt.tight_layout()
        return plt

    def save_test_results(self, results: Dict, filename: str = None):
        """Test sonuçlarını JSON olarak kaydeder"""
        if not filename:
            filename = f"performance_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.reporter.results_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        return filepath

    def load_test_results(self, filename: str) -> Dict:
        """Kaydedilmiş test sonuçlarını yükler"""
        filepath = os.path.join(self.reporter.results_dir, filename)
        with open(filepath, 'r') as f:
            return json.load(f)