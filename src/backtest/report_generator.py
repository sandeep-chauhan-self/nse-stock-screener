"""
Advanced Backtesting Report Generation System for FS.6 Compliance

This module provides comprehensive report generation capabilities for backtesting
results, creating institutional-quality reports with detailed analytics, charts,
and professional formatting. Supports multiple output formats including HTML,
PDF, and Excel.

Key Features:
- Executive summary with key performance indicators
- Detailed analytics tables and charts
- Risk analysis and drawdown reports
- Trade-level analysis and attribution
- Sector and time-based performance breakdown
- Professional HTML/PDF report generation
- Interactive charts and visualizations

Author: Enhanced Backtesting System
Created: 2025-01-20
"""

import os
import json
import warnings
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import seaborn as sns
from jinja2 import Template, Environment, FileSystemLoader

# Constants for repeated strings
DRAWDOWN_PERCENT = 'Drawdown (%)'
TRADE_ANALYSIS = 'Trade Analysis'

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Import our backtest modules
from .persistence import BacktestPersistence, BacktestMetadata, TradeRecord
from .performance_metrics import PerformanceCalculator, PerformanceMetrics
from .execution_engine import ExecutionEngine


class ReportFormat(Enum):
    """Report output formats"""
    HTML = "html"
    PDF = "pdf"
    EXCEL = "excel"
    JSON = "json"


class ChartType(Enum):
    """Available chart types for reports"""
    EQUITY_CURVE = "equity_curve"
    DRAWDOWN = "drawdown"
    MONTHLY_RETURNS = "monthly_returns"
    ROLLING_SHARPE = "rolling_sharpe"
    SECTOR_ATTRIBUTION = "sector_attribution"
    TRADE_DISTRIBUTION = "trade_distribution"
    UNDERWATER_CHART = "underwater_chart"
    RETURN_DISTRIBUTION = "return_distribution"


@dataclass
class ReportConfig:
    """Configuration for report generation"""
    title: str = "Backtesting Report"
    subtitle: str = "Comprehensive Performance Analysis"
    include_charts: bool = True
    include_trade_detail: bool = True
    include_sector_analysis: bool = True
    chart_width: int = 12
    chart_height: int = 6
    chart_dpi: int = 300
    color_scheme: str = "professional"  # professional, colorful, monochrome
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ReportSection:
    """Individual report section"""
    title: str
    content: str
    charts: List[str]
    tables: List[Dict[str, Any]]
    order: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class ChartGenerator:
    """Generates charts for backtesting reports"""
    
    def __init__(self, config: ReportConfig):
        self.config = config
        self.setup_style()
    
    def setup_style(self) -> None:
        """Setup matplotlib style based on configuration"""
        plt.style.use('seaborn-v0_8')
        
        # Color schemes
        if self.config.color_scheme == "professional":
            self.colors = {
                'primary': '#1f77b4',
                'secondary': '#ff7f0e',
                'success': '#2ca02c',
                'danger': '#d62728',
                'warning': '#ff7f0e',
                'info': '#17a2b8',
                'background': '#f8f9fa'
            }
        elif self.config.color_scheme == "colorful":
            self.colors = {
                'primary': '#3498db',
                'secondary': '#e74c3c',
                'success': '#2ecc71',
                'danger': '#e74c3c',
                'warning': '#f39c12',
                'info': '#9b59b6',
                'background': '#ecf0f1'
            }
        else:  # monochrome
            self.colors = {
                'primary': '#2c3e50',
                'secondary': '#7f8c8d',
                'success': '#27ae60',
                'danger': '#e74c3c',
                'warning': '#f39c12',
                'info': '#34495e',
                'background': '#bdc3c7'
            }
    
    def create_equity_curve_chart(self, equity_data: pd.DataFrame, 
                                  benchmark_data: Optional[pd.DataFrame] = None) -> str:
        """Create equity curve chart"""
        _, ax = plt.subplots(figsize=(self.config.chart_width, self.config.chart_height))
        
        # Plot strategy equity curve
        ax.plot(equity_data.index, equity_data['equity'], 
                color=self.colors['primary'], linewidth=2, label='Strategy')
        
        # Plot benchmark if provided
        if benchmark_data is not None:
            ax.plot(benchmark_data.index, benchmark_data['equity'],
                    color=self.colors['secondary'], linewidth=1.5, 
                    label='Benchmark', alpha=0.7)
        
        ax.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value (₹)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, p: f'₹{x:,.0f}'))
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save chart
        chart_path = f"equity_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_path, dpi=self.config.chart_dpi, bbox_inches='tight')
        plt.close()
        
        return chart_path
    
    def create_drawdown_chart(self, equity_data: pd.DataFrame) -> str:
        """Create drawdown chart"""
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.config.chart_width, 
                                                       self.config.chart_height * 1.2))
        
        # Calculate drawdown
        rolling_max = equity_data['equity'].expanding().max()
        drawdown = (equity_data['equity'] - rolling_max) / rolling_max * 100
        
        # Plot equity curve
        ax1.plot(equity_data.index, equity_data['equity'], 
                 color=self.colors['primary'], linewidth=2)
        ax1.fill_between(equity_data.index, equity_data['equity'], rolling_max,
                         alpha=0.3, color=self.colors['danger'])
        ax1.set_title('Equity Curve with Drawdowns', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value (₹)')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, p: f'₹{x:,.0f}'))
        
        # Plot drawdown
        ax2.fill_between(drawdown.index, drawdown, 0, 
                         color=self.colors['danger'], alpha=0.7)
        ax2.plot(drawdown.index, drawdown, color=self.colors['danger'], linewidth=1)
        ax2.set_title(DRAWDOWN_PERCENT, fontsize=12)
        ax2.set_xlabel('Date')
        ax2.set_ylabel(DRAWDOWN_PERCENT)
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis for both subplots
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save chart
        chart_path = f"drawdown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_path, dpi=self.config.chart_dpi, bbox_inches='tight')
        plt.close()
        
        return chart_path
    
    def create_monthly_returns_heatmap(self, returns: pd.Series) -> str:
        """Create monthly returns heatmap"""
        # Prepare monthly returns data
        monthly_returns = returns.resample('M').apply(
            lambda x: (1 + x).prod() - 1) * 100
        
        # Create pivot table for heatmap
        monthly_data = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values
        })
        
        pivot_data = monthly_data.pivot(index='Year', columns='Month', values='Return')
        
        # Create heatmap
        _, ax = plt.subplots(figsize=(self.config.chart_width, 
                                        self.config.chart_height * 0.8))
        
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn',
                    center=0, ax=ax, cbar_kws={'label': 'Return (%)'})
        
        ax.set_title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Month')
        ax.set_ylabel('Year')
        
        # Set month labels
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticklabels(month_labels)
        
        plt.tight_layout()
        
        # Save chart
        chart_path = f"monthly_returns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_path, dpi=self.config.chart_dpi, bbox_inches='tight')
        plt.close()
        
        return chart_path
    
    def create_rolling_metrics_chart(self, returns: pd.Series, 
                                     window: int = 252) -> str:
        """Create rolling metrics chart"""
        _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, 
                                                      figsize=(self.config.chart_width * 1.2, 
                                                               self.config.chart_height * 1.2))
        
        # Rolling Sharpe ratio
        rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
        ax1.plot(rolling_sharpe.index, rolling_sharpe, color=self.colors['primary'])
        ax1.set_title('Rolling Sharpe Ratio (1Y)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=1, color=self.colors['secondary'], linestyle='--', alpha=0.7)
        
        # Rolling volatility
        rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100
        ax2.plot(rolling_vol.index, rolling_vol, color=self.colors['warning'])
        ax2.set_title('Rolling Volatility (1Y)', fontsize=12)
        ax2.set_ylabel('Volatility (%)')
        ax2.grid(True, alpha=0.3)
        
        # Rolling max drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.rolling(window, min_periods=1).max()
        rolling_dd = ((cumulative_returns - rolling_max) / rolling_max * 100).rolling(window).min()
        ax3.plot(rolling_dd.index, rolling_dd, color=self.colors['danger'])
        ax3.set_title('Rolling Max Drawdown (1Y)', fontsize=12)
        ax3.set_ylabel(DRAWDOWN_PERCENT)
        ax3.grid(True, alpha=0.3)
        
        # Rolling Calmar ratio
        rolling_calmar = (returns.rolling(window).mean() * 252) / abs(rolling_dd / 100)
        rolling_calmar = rolling_calmar.replace([np.inf, -np.inf], np.nan)
        ax4.plot(rolling_calmar.index, rolling_calmar, color=self.colors['success'])
        ax4.set_title('Rolling Calmar Ratio (1Y)', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        # Format all axes
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save chart
        chart_path = f"rolling_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_path, dpi=self.config.chart_dpi, bbox_inches='tight')
        plt.close()
        
        return chart_path
    
    def create_trade_distribution_chart(self, trades: List[TradeRecord]) -> str:
        """Create trade distribution analysis chart"""
        if not trades:
            return ""
        
        # Extract trade data
        trade_returns = [trade.pnl_percent for trade in trades]
        trade_durations = [(trade.exit_time - trade.entry_time).days 
                          for trade in trades if trade.exit_time]
        
        _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, 
                                                      figsize=(self.config.chart_width * 1.2, 
                                                               self.config.chart_height * 1.2))
        
        # Trade returns histogram
        ax1.hist(trade_returns, bins=30, alpha=0.7, color=self.colors['primary'],
                 edgecolor='black')
        ax1.axvline(x=0, color=self.colors['danger'], linestyle='--', alpha=0.7)
        ax1.set_title('Trade Returns Distribution', fontsize=12)
        ax1.set_xlabel('Return (%)')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Trade duration histogram
        if trade_durations:
            ax2.hist(trade_durations, bins=20, alpha=0.7, color=self.colors['secondary'],
                     edgecolor='black')
            ax2.set_title('Trade Duration Distribution', fontsize=12)
            ax2.set_xlabel('Duration (Days)')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
        
        # Win/Loss ratio pie chart
        winning_trades = len([r for r in trade_returns if r > 0])
        losing_trades = len([r for r in trade_returns if r < 0])
        breakeven_trades = len([r for r in trade_returns if r == 0])
        
        labels = ['Winning', 'Losing', 'Breakeven']
        sizes = [winning_trades, losing_trades, breakeven_trades]
        colors = [self.colors['success'], self.colors['danger'], self.colors['info']]
        
        ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Win/Loss Distribution', fontsize=12)
        
        # Monthly trade count
        trade_months = pd.Series([trade.entry_time.to_pydatetime().strftime('%Y-%m') 
                                 for trade in trades]).value_counts().sort_index()
        ax4.bar(range(len(trade_months)), trade_months.values, 
                color=self.colors['info'], alpha=0.7)
        ax4.set_title('Monthly Trade Count', fontsize=12)
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Number of Trades')
        ax4.set_xticks(range(0, len(trade_months), max(1, len(trade_months)//6)))
        ax4.set_xticklabels([trade_months.index[i] for i in 
                            range(0, len(trade_months), max(1, len(trade_months)//6))],
                           rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save chart
        chart_path = f"trade_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_path, dpi=self.config.chart_dpi, bbox_inches='tight')
        plt.close()
        
        return chart_path


class HTMLReportGenerator:
    """Generates HTML reports using Jinja2 templates"""
    
    def __init__(self, config: ReportConfig):
        self.config = config
        self.template = self._create_template()
    
    def _create_template(self) -> Template:
        """Create Jinja2 template for HTML report"""
        template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ config.title }}</title>
    <style>
        body { 
            font-family: 'Segoe UI', Arial, sans-serif; 
            line-height: 1.6; 
            margin: 0; 
            padding: 20px;
            background-color: #f8f9fa;
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header { 
            text-align: center; 
            margin-bottom: 30px; 
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 20px;
        }
        .header h1 { 
            color: #2c3e50; 
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .header h2 { 
            color: #7f8c8d; 
            margin-top: 0;
            font-weight: normal;
        }
        .section { 
            margin-bottom: 40px; 
        }
        .section h3 { 
            color: #2c3e50; 
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-bottom: 20px;
        }
        .metrics-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
            gap: 20px; 
            margin-bottom: 30px;
        }
        .metric-card { 
            background: #f8f9fa; 
            padding: 20px; 
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }
        .metric-card h4 { 
            margin: 0 0 10px 0; 
            color: #2c3e50;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .metric-card .value { 
            font-size: 1.8em; 
            font-weight: bold; 
            color: #2c3e50;
        }
        .positive { color: #27ae60 !important; }
        .negative { color: #e74c3c !important; }
        .chart { 
            text-align: center; 
            margin: 20px 0;
        }
        .chart img { 
            max-width: 100%; 
            height: auto;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        table { 
            width: 100%; 
            border-collapse: collapse; 
            margin: 20px 0;
            background: white;
        }
        th, td { 
            padding: 12px; 
            text-align: left; 
            border-bottom: 1px solid #ddd;
        }
        th { 
            background-color: #f8f9fa; 
            font-weight: bold;
            color: #2c3e50;
        }
        tr:nth-child(even) { 
            background-color: #f8f9fa; 
        }
        .footer { 
            margin-top: 50px; 
            text-align: center; 
            color: #7f8c8d;
            border-top: 1px solid #e9ecef;
            padding-top: 20px;
        }
        .summary-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
        }
        .summary-box h3 {
            margin-top: 0;
            border: none;
            padding: 0;
            color: white;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ config.title }}</h1>
            <h2>{{ config.subtitle }}</h2>
            <p>Generated on {{ generation_date }}</p>
        </div>

        <!-- Executive Summary -->
        <div class="section">
            <div class="summary-box">
                <h3>Executive Summary</h3>
                <p>{{ executive_summary }}</p>
            </div>
        </div>

        <!-- Key Metrics -->
        <div class="section">
            <h3>Key Performance Indicators</h3>
            <div class="metrics-grid">
                {% for metric in key_metrics %}
                <div class="metric-card">
                    <h4>{{ metric.name }}</h4>
                    <div class="value {{ 'positive' if metric.value > 0 else 'negative' if metric.value < 0 else '' }}">
                        {{ metric.formatted_value }}
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>

        <!-- Charts Section -->
        {% if charts %}
        <div class="section">
            <h3>Performance Charts</h3>
            {% for chart in charts %}
            <div class="chart">
                <h4>{{ chart.title }}</h4>
                <img src="{{ chart.path }}" alt="{{ chart.title }}">
            </div>
            {% endfor %}
        </div>
        {% endif %}

        <!-- Detailed Metrics -->
        {% for section in sections %}
        <div class="section">
            <h3>{{ section.title }}</h3>
            {{ section.content | safe }}
            
            {% if section.tables %}
            {% for table in section.tables %}
            <h4>{{ table.title }}</h4>
            <table>
                <thead>
                    <tr>
                        {% for header in table.headers %}
                        <th>{{ header }}</th>
                        {% endfor %}
                    </tr>
                </thead>
                <tbody>
                    {% for row in table.rows %}
                    <tr>
                        {% for cell in row %}
                        <td>{{ cell }}</td>
                        {% endfor %}
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% endfor %}
            {% endif %}
        </div>
        {% endfor %}

        <div class="footer">
            <p>Report generated by Enhanced Backtesting System v1.0</p>
            <p>© 2025 Advanced Trading Analytics</p>
        </div>
    </div>
</body>
</html>
        """
        return Template(template_str)
    
    def generate_report(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML report from data"""
        return self.template.render(**report_data)


class BacktestReportGenerator:
    """Main class for generating comprehensive backtesting reports"""
    
    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()
        self.chart_generator = ChartGenerator(self.config)
        self.html_generator = HTMLReportGenerator(self.config)
        self.charts_created = []
    
    def generate_comprehensive_report(self, 
                                      backtest_id: str,
                                      persistence: BacktestPersistence,
                                      output_dir: str = "output/reports",
                                      formats: List[ReportFormat] = None) -> Dict[str, str]:
        """
        Generate comprehensive backtest report
        
        Args:
            backtest_id: ID of the backtest to report on
            persistence: BacktestPersistence instance
            output_dir: Output directory for reports
            formats: List of formats to generate
            
        Returns:
            Dictionary mapping format to file path
        """
        if formats is None:
            formats = [ReportFormat.HTML]
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Load backtest data
            metadata = persistence.load_metadata(backtest_id)
            trades = persistence.load_trades(backtest_id)
            equity_curve = persistence.load_equity_curve(backtest_id)
            
            # Calculate comprehensive metrics
            if not equity_curve.empty:
                returns = equity_curve['equity'].pct_change().dropna()
                performance_calc = PerformanceCalculator()
                metrics = performance_calc.calculate_comprehensive_metrics(
                    returns, trades, equity_curve['equity']
                )
            else:
                metrics = PerformanceMetrics()
                returns = pd.Series(dtype=float)
            
            # Generate charts if requested
            charts = []
            if self.config.include_charts and not equity_curve.empty:
                charts = self._generate_all_charts(equity_curve, returns, trades)
            
            # Prepare report data
            report_data = self._prepare_report_data(
                metadata, metrics, trades, charts
            )
            
            # Generate reports in requested formats
            generated_reports = {}
            
            for report_format in formats:
                if report_format == ReportFormat.HTML:
                    file_path = self._generate_html_report(report_data, output_dir, backtest_id)
                    generated_reports[ReportFormat.HTML] = file_path
                elif report_format == ReportFormat.EXCEL:
                    file_path = self._generate_excel_report(report_data, output_dir, backtest_id)
                    generated_reports[ReportFormat.EXCEL] = file_path
                elif report_format == ReportFormat.JSON:
                    file_path = self._generate_json_report(report_data, output_dir, backtest_id)
                    generated_reports[ReportFormat.JSON] = file_path
            
            return generated_reports
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate report: {str(e)}") from e
    
    def _generate_all_charts(self, equity_curve: pd.DataFrame, 
                           returns: pd.Series, trades: List[TradeRecord]) -> List[Dict[str, str]]:
        """Generate all charts for the report"""
        charts = []
        
        try:
            # Equity curve chart
            chart_path = self.chart_generator.create_equity_curve_chart(equity_curve)
            charts.append({
                'title': 'Equity Curve',
                'path': chart_path,
                'type': ChartType.EQUITY_CURVE.value
            })
            self.charts_created.append(chart_path)
            
            # Drawdown chart
            chart_path = self.chart_generator.create_drawdown_chart(equity_curve)
            charts.append({
                'title': 'Drawdown Analysis',
                'path': chart_path,
                'type': ChartType.DRAWDOWN.value
            })
            self.charts_created.append(chart_path)
            
            # Monthly returns heatmap
            if len(returns) > 30:  # Only if we have enough data
                chart_path = self.chart_generator.create_monthly_returns_heatmap(returns)
                charts.append({
                    'title': 'Monthly Returns Heatmap',
                    'path': chart_path,
                    'type': ChartType.MONTHLY_RETURNS.value
                })
                self.charts_created.append(chart_path)
            
            # Rolling metrics
            if len(returns) > 252:  # Only if we have at least 1 year of data
                chart_path = self.chart_generator.create_rolling_metrics_chart(returns)
                charts.append({
                    'title': 'Rolling Performance Metrics',
                    'path': chart_path,
                    'type': ChartType.ROLLING_SHARPE.value
                })
                self.charts_created.append(chart_path)
            
            # Trade distribution
            if trades:
                chart_path = self.chart_generator.create_trade_distribution_chart(trades)
                charts.append({
                    'title': TRADE_ANALYSIS,
                    'path': chart_path,
                    'type': ChartType.TRADE_DISTRIBUTION.value
                })
                self.charts_created.append(chart_path)
                
        except Exception as e:
            print(f"Warning: Error generating charts: {e}")
        
        return charts
    
    def _prepare_report_data(self, metadata: BacktestMetadata, 
                           metrics: PerformanceMetrics,
                           trades: List[TradeRecord],
                           charts: List[Dict[str, str]]) -> Dict[str, Any]:
        """Prepare data for report generation"""
        
        # Executive summary
        executive_summary = self._generate_executive_summary(metadata, metrics, trades)
        
        # Key metrics for dashboard
        key_metrics = [
            {
                'name': 'Total Return',
                'value': metrics.total_return,
                'formatted_value': f"{metrics.total_return:.2%}"
            },
            {
                'name': 'Annual Return',
                'value': metrics.annual_return,
                'formatted_value': f"{metrics.annual_return:.2%}"
            },
            {
                'name': 'Sharpe Ratio',
                'value': metrics.sharpe_ratio,
                'formatted_value': f"{metrics.sharpe_ratio:.2f}"
            },
            {
                'name': 'Max Drawdown',
                'value': -abs(metrics.max_drawdown),
                'formatted_value': f"{metrics.max_drawdown:.2%}"
            },
            {
                'name': 'Win Rate',
                'value': metrics.win_rate,
                'formatted_value': f"{metrics.win_rate:.1%}"
            },
            {
                'name': 'Total Trades',
                'value': len(trades),
                'formatted_value': f"{len(trades):,}"
            }
        ]
        
        # Detailed sections
        sections = [
            self._create_performance_section(metrics),
            self._create_risk_section(metrics),
            self._create_trade_analysis_section(trades, metrics)
        ]
        
        return {
            'config': self.config,
            'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'executive_summary': executive_summary,
            'key_metrics': key_metrics,
            'charts': charts,
            'sections': sections,
            'metadata': metadata,
            'backtest_period': f"{metadata.start_date} to {metadata.end_date}"
        }
    
    def _generate_executive_summary(self, metadata: BacktestMetadata,
                                  metrics: PerformanceMetrics,
                                  trades: List[TradeRecord]) -> str:
        """Generate executive summary text"""
        summary_parts = []
        
        # Strategy overview
        summary_parts.append(
            f"This backtest analyzed the '{metadata.strategy_name}' strategy "
            f"from {metadata.start_date.strftime('%B %Y')} to {metadata.end_date.strftime('%B %Y')}."
        )
        
        # Performance summary
        if metrics.total_return >= 0:
            performance_desc = "generated positive returns"
        else:
            performance_desc = "experienced losses"
        
        summary_parts.append(
            f"The strategy {performance_desc} of {metrics.total_return:.1%} "
            f"with an annualized return of {metrics.annual_return:.1%}."
        )
        
        # Risk assessment
        if metrics.sharpe_ratio > 1.0:
            risk_desc = "demonstrated excellent risk-adjusted performance"
        elif metrics.sharpe_ratio > 0.5:
            risk_desc = "showed reasonable risk-adjusted returns"
        else:
            risk_desc = "exhibited poor risk-adjusted performance"
        
        summary_parts.append(
            f"The strategy {risk_desc} with a Sharpe ratio of {metrics.sharpe_ratio:.2f} "
            f"and maximum drawdown of {metrics.max_drawdown:.1%}."
        )
        
        # Trading activity
        if trades:
            win_rate = metrics.win_rate
            summary_parts.append(
                f"Over {len(trades)} trades, the strategy achieved a {win_rate:.1%} win rate."
            )
        
        return " ".join(summary_parts)
    
    def _create_performance_section(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Create performance metrics section"""
        content = "<p>Comprehensive performance analysis of the trading strategy.</p>"
        
        # Performance metrics table
        performance_table = {
            'title': 'Performance Metrics',
            'headers': ['Metric', 'Value'],
            'rows': [
                ['Total Return', f"{metrics.total_return:.2%}"],
                ['Annual Return', f"{metrics.annual_return:.2%}"],
                ['Volatility', f"{metrics.volatility:.2%}"],
                ['Sharpe Ratio', f"{metrics.sharpe_ratio:.2f}"],
                ['Sortino Ratio', f"{metrics.sortino_ratio:.2f}"],
                ['Calmar Ratio', f"{metrics.calmar_ratio:.2f}"],
                ['Information Ratio', f"{metrics.information_ratio:.2f}"]
            ]
        }
        
        return {
            'title': 'Performance Analysis',
            'content': content,
            'tables': [performance_table],
            'charts': [],
            'order': 1
        }
    
    def _create_risk_section(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Create risk analysis section"""
        content = "<p>Detailed risk assessment and drawdown analysis.</p>"
        
        # Risk metrics table
        risk_table = {
            'title': 'Risk Metrics',
            'headers': ['Metric', 'Value'],
            'rows': [
                ['Maximum Drawdown', f"{metrics.max_drawdown:.2%}"],
                ['Value at Risk (95%)', f"{metrics.var_95:.2%}"],
                ['Expected Shortfall (95%)', f"{metrics.expected_shortfall:.2%}"],
                ['Downside Deviation', f"{metrics.downside_deviation:.2%}"],
                ['Up Capture Ratio', f"{metrics.up_capture:.2f}"],
                ['Down Capture Ratio', f"{metrics.down_capture:.2f}"],
                ['Beta', f"{metrics.beta:.2f}"]
            ]
        }
        
        return {
            'title': 'Risk Analysis',
            'content': content,
            'tables': [risk_table],
            'charts': [],
            'order': 2
        }
    
    def _create_trade_analysis_section(self, trades: List[TradeRecord],
                                     metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Create trade analysis section"""
        if not trades:
            return {
                'title': TRADE_ANALYSIS,
                'content': "<p>No trades available for analysis.</p>",
                'tables': [],
                'charts': [],
                'order': 3
            }
        
        content = f"<p>Analysis of {len(trades)} trades executed during the backtest period.</p>"
        
        # Trade statistics table
        winning_trades = [t for t in trades if t.pnl_percent > 0]
        losing_trades = [t for t in trades if t.pnl_percent < 0]
        
        avg_win = np.mean([t.pnl_percent for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl_percent for t in losing_trades]) if losing_trades else 0
        
        trade_table = {
            'title': 'Trade Statistics',
            'headers': ['Metric', 'Value'],
            'rows': [
                ['Total Trades', f"{len(trades):,}"],
                ['Winning Trades', f"{len(winning_trades):,}"],
                ['Losing Trades', f"{len(losing_trades):,}"],
                ['Win Rate', f"{metrics.win_rate:.1%}"],
                ['Average Win', f"{avg_win:.2%}"],
                ['Average Loss', f"{avg_loss:.2%}"],
                ['Profit Factor', f"{metrics.profit_factor:.2f}"],
                ['Average Trade', f"{np.mean([t.pnl_percent for t in trades]):.2%}"]
            ]
        }
        
        return {
            'title': TRADE_ANALYSIS,
            'content': content,
            'tables': [trade_table],
            'charts': [],
            'order': 3
        }
    
    def _generate_html_report(self, report_data: Dict[str, Any], 
                            output_dir: str, backtest_id: str) -> str:
        """Generate HTML report"""
        html_content = self.html_generator.generate_report(report_data)
        
        filename = f"backtest_report_{backtest_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        file_path = os.path.join(output_dir, filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return file_path
    
    def _generate_excel_report(self, report_data: Dict[str, Any],
                             output_dir: str, backtest_id: str) -> str:
        """Generate Excel report"""
        filename = f"backtest_report_{backtest_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        file_path = os.path.join(output_dir, filename)
        
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = []
            for metric in report_data['key_metrics']:
                summary_data.append([metric['name'], metric['formatted_value']])
            
            summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Detailed metrics sheets
            for section in report_data['sections']:
                if section['tables']:
                    for table in section['tables']:
                        table_df = pd.DataFrame(table['rows'], columns=table['headers'])
                        sheet_name = table['title'][:31]  # Excel sheet name limit
                        table_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        return file_path
    
    def _generate_json_report(self, report_data: Dict[str, Any],
                            output_dir: str, backtest_id: str) -> str:
        """Generate JSON report"""
        filename = f"backtest_report_{backtest_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        file_path = os.path.join(output_dir, filename)
        
        # Prepare JSON-serializable data
        json_data = {
            'metadata': {
                'backtest_id': backtest_id,
                'generation_date': report_data['generation_date'],
                'strategy_name': report_data.get('metadata', {}).get('strategy_name', 'Unknown'),
                'backtest_period': report_data.get('backtest_period', 'Unknown')
            },
            'executive_summary': report_data['executive_summary'],
            'key_metrics': report_data['key_metrics'],
            'detailed_sections': report_data['sections']
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        return file_path
    
    def cleanup_charts(self) -> None:
        """Clean up generated chart files"""
        for chart_path in self.charts_created:
            try:
                if os.path.exists(chart_path):
                    os.remove(chart_path)
            except Exception as e:
                print(f"Warning: Could not remove chart file {chart_path}: {e}")
        self.charts_created.clear()


def demo_report_generation():
    """Demonstration of report generation capabilities"""
    print("🚀 Enhanced Backtesting Report Generator Demo")
    print("=" * 50)
    
    # Demo configuration
    config = ReportConfig(
        title="NSE Stock Strategy Backtest",
        subtitle="Enhanced Performance Analysis Report",
        include_charts=True,
        include_trade_detail=True,
        include_sector_analysis=True,
        color_scheme="professional"
    )
    
    print(f"Report Configuration: {config.title}")
    print(f"Charts Enabled: {config.include_charts}")
    print(f"Color Scheme: {config.color_scheme}")
    
    # Initialize report generator
    report_gen = BacktestReportGenerator(config)
    
    print("\n✅ Report Generator initialized successfully")
    print("📊 Ready to generate comprehensive backtest reports")
    print("\nFeatures available:")
    print("- Executive summary with key insights")
    print("- Interactive charts and visualizations")
    print("- Detailed performance metrics")
    print("- Risk analysis and drawdown reports")
    print("- Trade-level analysis")
    print("- Multiple output formats (HTML, Excel, JSON)")
    
    return report_gen


if __name__ == "__main__":
    demo_report_generation()