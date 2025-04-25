import pandas as pd
import numpy as np
import quantstats as qs
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from typing import Dict, List, Optional
import json
import os
import webbrowser
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
import pdfkit
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ReportGenerator:
    def __init__(self, results_dir: str = 'backtest_results'):
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        self.env = Environment(loader=FileSystemLoader('templates'))

    def generate_html_report(self, backtest_results: Dict, strategy_name: str) -> str:
        """QuantStats tabanlı interaktif HTML raporu oluşturur"""
        returns = pd.Series(backtest_results['returns'])
        returns.index = pd.to_datetime(returns.index)
        
        # HTML raporu oluştur
        report_path = os.path.join(self.results_dir, f'{strategy_name}_report.html')
        qs.reports.html(
            returns,
            output=report_path,
            title=f'{strategy_name} Backtest Report',
            download_filename=report_path
        )
        return report_path

    def generate_pdf_report(self, backtest_results: Dict, strategy_name: str) -> str:
        """PDF formatında detaylı rapor oluşturur"""
        # Verileri hazırla
        metrics = self._calculate_metrics(backtest_results)
        trades_df = pd.DataFrame(backtest_results['trades'])
        equity_curve = backtest_results['equity_curve']
        
        # Grafikleri oluştur
        fig = self._create_composite_figure(equity_curve, trades_df, metrics)
        
        # HTML şablonunu render et
        template = self.env.get_template('report_template.html')
        html_content = template.render(
            strategy_name=strategy_name,
            date=datetime.now().strftime('%Y-%m-%d'),
            metrics=metrics,
            trades=trades_df.to_dict('records')[:50],  # Son 50 işlem
            win_rate=metrics['win_rate'],
            sharpe=metrics['sharpe_ratio'],
            mdd=metrics['max_drawdown']
        )
        
        # PDF'e dönüştür
        report_path = os.path.join(self.results_dir, f'{strategy_name}_report.pdf')
        pdfkit.from_string(html_content, report_path, options={
            'encoding': 'UTF-8',
            'page-size': 'A3',
            'orientation': 'Landscape'
        })
        
        return report_path

    def _calculate_metrics(self, results: Dict) -> Dict:
        """Temel performans metriklerini hesaplar"""
        returns = pd.Series(results['returns'])
        trades = pd.DataFrame(results['trades'])
        
        # Win rate hesapla
        win_rate = len(trades[trades['pnl'] > 0]) / len(trades) if len(trades) > 0 else 0
        
        # Sharpe oranı (yıllık)
        sharpe = np.sqrt(365) * returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Maksimum çöküş
        equity = (1 + returns).cumprod()
        peak = equity.expanding(min_periods=1).max()
        drawdown = (equity - peak) / peak
        max_drawdown = drawdown.min()
        
        return {
            'total_return': equity.iloc[-1] - 1,
            'annualized_return': (equity.iloc[-1] ** (365/len(returns))) - 1,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'profit_factor': -trades[trades['pnl'] > 0]['pnl'].sum() / 
                            trades[trades['pnl'] < 0]['pnl'].sum() if trades[trades['pnl'] < 0]['pnl'].sum() < 0 else float('inf'),
            'avg_trade': trades['pnl'].mean(),
            'total_trades': len(trades)
        }

    def _create_composite_figure(self, equity_curve: pd.Series, 
                               trades_df: pd.DataFrame,
                               metrics: Dict) -> plt.Figure:
        """Bileşik performans grafiği oluşturur"""
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 2, figure=fig)
        
        # 1. Equity Curve
        ax1 = fig.add_subplot(gs[0, :])
        equity_curve.plot(ax=ax1, title='Equity Curve', color='blue')
        ax1.axhline(1, linestyle='--', color='gray')
        ax1.set_ylabel('Portfolio Value')
        
        # 2. Drawdown
        ax2 = fig.add_subplot(gs[1, :])
        (equity_curve / equity_curve.cummax() - 1).plot(
            ax=ax2, title='Drawdown', color='red')
        ax2.fill_between(equity_curve.index, 0, 
                        (equity_curve / equity_curve.cummax() - 1), 
                        color='red', alpha=0.3)
        ax2.set_ylabel('Drawdown')
        
        # 3. Monthly Returns
        ax3 = fig.add_subplot(gs[2, 0])
        monthly_returns = equity_curve.resample('M').last().pct_change()
        sns.heatmap(
            pd.DataFrame(monthly_returns.values.reshape(-1, 1), 
            annot=True, fmt=".1%", cmap="RdYlGn", 
            ax=ax3, cbar=False)
        ax3.set_title('Monthly Returns')
        
        # 4. Trade Analysis
        ax4 = fig.add_subplot(gs[2, 1])
        trades_df['pnl'].hist(ax=ax4, bins=30, color='green', alpha=0.7)
        ax4.axvline(0, color='red', linestyle='--')
        ax4.set_title('Trade PnL Distribution')
        
        plt.tight_layout()
        return fig

    def generate_interactive_report(self, backtest_results: Dict, 
                                  strategy_name: str) -> str:
        """Plotly ile interaktif rapor oluşturur"""
        returns = pd.Series(backtest_results['returns'])
        trades = pd.DataFrame(backtest_results['trades'])
        equity = (1 + returns).cumprod()
        
        # Ana figür
        fig = make_subplots(rows=3, cols=2, 
                           specs=[[{"colspan": 2}, None],
                                 [{"colspan": 2}, None],
                                 [{}, {}]],
                           subplot_titles=("Equity Curve", 
                                          "Drawdown", 
                                          "Monthly Returns", 
                                          "Trade Distribution"))
        
        # Equity Curve
        fig.add_trace(
            go.Scatter(x=equity.index, y=equity, 
                      name="Portfolio Value"),
            row=1, col=1
        )
        
        # Drawdown
        drawdown = (equity / equity.cummax() - 1)
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown, 
                      fill='tozeroy', 
                      fillcolor='rgba(255,0,0,0.3)',
                      line=dict(color='red'),
                      name="Drawdown"),
            row=2, col=1
        )
        
        # Monthly Returns
        monthly_returns = equity.resample('M').last().pct_change()
        monthly_matrix = monthly_returns.values.reshape(-1, 1)
        fig.add_trace(
            go.Heatmap(z=monthly_matrix,
                      colorscale='RdYlGn',
                      text=[[f"{x:.1%}" for x in row] for row in monthly_matrix],
                      texttemplate="%{text}",
                      showscale=False),
            row=3, col=1
        )
        
        # Trade Distribution
        fig.add_trace(
            go.Histogram(x=trades['pnl'],
                        nbinsx=30,
                        marker_color='green',
                        name='Trade PnL'),
            row=3, col=2
        )
        
        # Layout ayarları
        fig.update_layout(
            height=1000,
            title_text=f"{strategy_name} - Backtest Results",
            showlegend=True
        )
        
        # HTML olarak kaydet
        report_path = os.path.join(self.results_dir, f'{strategy_name}_interactive.html')
        fig.write_html(report_path)
        return report_path

    def save_backtest_results(self, results: Dict, strategy_name: str) -> str:
        """Backtest sonuçlarını JSON olarak kaydeder"""
        filename = f"{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = os.path.join(self.results_dir, filename)
        
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        return report_path

    def open_report_in_browser(self, report_path: str):
        """Raporu varsayılan tarayıcıda açar"""
        webbrowser.open(f'file://{os.path.abspath(report_path)}')

    def generate_all_reports(self, backtest_results: Dict, strategy_name: str):
        """Tüm rapor formatlarını oluşturur"""
        reports = {
            'html': self.generate_html_report(backtest_results, strategy_name),
            'pdf': self.generate_pdf_report(backtest_results, strategy_name),
            'interactive': self.generate_interactive_report(backtest_results, strategy_name),
            'json': self.save_backtest_results(backtest_results, strategy_name)
        }
        
        # Raporları aç
        self.open_report_in_browser(reports['html'])
        return reports