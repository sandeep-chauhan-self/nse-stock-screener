import logging
"""
Early Warning System for Potential Stock Jumps
Combines multiple signals for higher probability setups
"""
from datetime import datetime, timedelta
import argparse
import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf
class EarlyWarningSystem:
    def __init__(self, custom_stocks=None, input_file=None, batch_size=50, timeout=10) -> None:

        # Default NSE symbols that work with Yahoo Finance
        self.default_stocks = [

            # Large Cap
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
            'HINDUNILVR.NS', 'SBIN.NS', 'BAJFINANCE.NS', 'KOTAKBANK.NS', 'LT.NS',

            # Mid Cap
            'TATAMOTORS.NS', 'MARUTI.NS', 'M&M.NS', 'ADANIENT.NS', 'ADANIPORTS.NS',
            'EICHERMOT.NS', 'HEROMOTOCO.NS', 'INDUSINDBK.NS', 'SBILIFE.NS', 'BAJAJFINSV.NS',

            # IT Sector
            'WIPRO.NS', 'TECHM.NS', 'HCLTECH.NS', 'LTIM.NS', 'PERSISTENT.NS',

            # Pharma Sector
            'SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'LUPIN.NS',

            # Others
            'SAIL.NS', 'TATASTEEL.NS', 'BHARTIARTL.NS', 'AXISBANK.NS', 'GRASIM.NS'
        ]

        # Batch processing settings
        self.batch_size = batch_size
        self.timeout = timeout

        # Determine which stocks to use
        if input_file:

            # Load stocks from file
            self.nse_stocks = self.load_stocks_from_file(input_file)
            print(f"Loaded {len(self.nse_stocks)} stocks from file: {input_file}")
        elif custom_stocks:

            # Use provided List[str] of custom stocks
            self.nse_stocks = custom_stocks
            print(f"Using {len(self.nse_stocks)} custom stocks")
        else:

            # Use default List[str]
            self.nse_stocks = self.default_stocks
            print(f"Using {len(self.nse_stocks)} default stocks")
    def load_stocks_from_file(self, file_path):
        """Load stock symbols from a file (text or CSV)"""
        stocks = []
        try:

            # Check if file exists
            if not os.path.exists(file_path):
                logging.error(f"Error: File {file_path} not found")
                return self.default_stocks

            # Read file extension
            _, ext = os.path.splitext(file_path)
            if ext.lower() == '.csv':

                # Read CSV file
                df = pd.read_csv(file_path)

                # Check if 'Symbol' column exists
                if 'Symbol' in df.columns:
                    stocks = df['Symbol'].tolist()
                else:

                    # Use the first column
                    stocks = df.iloc[:, 0].tolist()
            else:

                # Read text file (one symbol per line)
                with open(file_path, 'r') as f:
                    stocks = [line.strip() for line in f if line.strip()]

            # Validate and format stock symbols
            formatted_stocks = []
            for stock in stocks:

                # Check if it already has .NS suffix
                if not stock.endswith('.NS') and stock.strip():
                    stock = f"{stock.strip()}.NS"
                formatted_stocks.append(stock)
            return formatted_stocks
        except Exception as e:
            logging.error(f"Error loading stocks from file: {e}")
            return self.default_stocks
    def unusual_volume_detector(self, symbol, volume_threshold=3):
        """Detect unusual volume patterns"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period="3mo")
            if data.empty or len(data) < 20:
                logging.warning(f"Warning: Insufficient data for {symbol}")
                return None

            # Calculate volume metrics
            avg_volume_20 = data['Volume'].rolling(20).mean().iloc[-2]
            current_volume = data['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume_20

            # Price change
            price_change = ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) /
                          data['Close'].iloc[-2]) * 100
            if volume_ratio >= volume_threshold:
                return {
                    'symbol': symbol,
                    'volume_ratio': round(volume_ratio, 2),
                    'price_change': round(price_change, 2),
                    'signal_strength': 'HIGH' if volume_ratio > 5 else 'MEDIUM'
                }
        except Exception as e:
            logging.error(f"Error in volume detection for {symbol}: {e}")
            return None
    def momentum_acceleration_detector(self, symbol):
        """Detect momentum acceleration patterns"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period="3mo")
            if data.empty or len(data) < 26:
                logging.warning(f"Warning: Insufficient data for {symbol}")
                return None

            # Calculate RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

            # Handle division by zero
            if loss.iloc[-1] == 0:
                current_rsi = 100
            else:
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1]

            # Calculate MACD
            exp1 = data['Close'].ewm(span=12).mean()
            exp2 = data['Close'].ewm(span=26).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9).mean()

            # Current values
            current_macd = macd.iloc[-1]
            current_signal = signal.iloc[-1]

            # Momentum score
            momentum_score = 0
            if 60 <= current_rsi <= 80:
  # Sweet spot for momentum
                momentum_score += 2
            if current_macd > current_signal:
  # MACD bullish
                momentum_score += 2
            if current_macd > 0:
  # MACD above zero
                momentum_score += 1
            return {
                'symbol': symbol,
                'rsi': round(current_rsi, 2),
                'macd_signal': 'BULLISH' if current_macd > current_signal else 'BEARISH',
                'momentum_score': momentum_score,
                'probability': 'HIGH' if momentum_score >= 4 else 'MEDIUM' if momentum_score >= 2 else 'LOW'
            }
        except Exception as e:
            logging.error(f"Error in momentum detection for {symbol}: {e}")
            return None
    def analyze_and_plot_stock(self, symbol):
        """Analyze and plot a stock chart with technical indicators"""
        try:
            print(f"\nAnalyzing {symbol} in detail...")
            stock = yf.Ticker(symbol)
            data = stock.history(period="6mo")
            if data.empty or len(data) < 50:
                print(f"Insufficient data for {symbol} to generate chart")
                return None

            # Create figure and axis
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1, 1]})

            # Price and volume plot
            ax1.plot(data.index, data['Close'], 'b-', label='Close Price')
            ax1.set_title(f'{symbol} - Momentum Analysis', fontsize=15)
            ax1.set_ylabel('Price', fontsize=12)
            ax1.grid(True, alpha=0.3)

            # Add moving averages
            data['MA20'] = data['Close'].rolling(window=20).mean()
            data['MA50'] = data['Close'].rolling(window=50).mean()
            ax1.plot(data.index, data['MA20'], 'r-', label='20-day MA')
            ax1.plot(data.index, data['MA50'], 'g-', label='50-day MA')

            # Add volume as bar chart at the bottom of price chart
            volume_ax = ax1.twinx()
            volume_ax.bar(data.index, data['Volume'], alpha=0.3, color='gray')
            volume_ax.set_ylabel('Volume', fontsize=12)
            volume_ax.set_ylim(0, data['Volume'].max() * 5)

            # Add legend
            ax1.legend(loc='upper left')

            # RSI plot
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            with np.errstate(divide='ignore', invalid='ignore'):
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
            ax2.plot(data.index, rsi, 'purple', label='RSI')
            ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            ax2.set_ylabel('RSI', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 100)

            # MACD plot
            exp1 = data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = data['Close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            ax3.plot(data.index, macd, 'b-', label='MACD')
            ax3.plot(data.index, signal, 'r-', label='Signal')
            ax3.bar(data.index, macd - signal, alpha=0.5, color=np.where(macd > signal, 'g', 'r'))
            ax3.set_ylabel('MACD', fontsize=12)
            ax3.grid(True, alpha=0.3)
            ax3.legend(loc='upper left')
            plt.tight_layout()

            # Create charts directory if it doesn't exist
            charts_dir = 'charts'
            if os.path.isabs(charts_dir):
                os.makedirs(charts_dir, exist_ok=True)
            else:
                os.makedirs(os.path.join(os.getcwd(), charts_dir), exist_ok=True)

            # Save the figure
            chart_path = os.path.join(charts_dir, f'{symbol.replace(".NS", "")}_chart.png')
            plt.savefig(chart_path)
            plt.close()
            print(f"Chart saved to {chart_path}")
            return chart_path
        except Exception as e:
            logging.error(f"Error generating chart for {symbol}: {e}")
            return None
    def backtest_signal(self, symbol, days=5):
        """Backtest a signal by looking at next N days performance"""
        try:
            print(f"\nBacktesting {symbol} for {days} days forward performance...")
            stock = yf.Ticker(symbol)

            # Get historical data including future periods if available
            data = stock.history(period="3mo")
            if data.empty or len(data) < 30:
                print(f"Insufficient data for {symbol} to perform backtest")
                return None

            # Find the last trading day in our data
            last_date = data.index[-1]

            # Get current price
            current_price = data['Close'].iloc[-1]

            # Get previous prices for the lookback period
            lookback_prices = []
            forward_prices = []

            # For demonstration, we'll look at past signals to simulate forward testing
            for i in range(1, min(30, len(data)-days)):

                # This would be our "signal day"
                signal_day = data.index[-(i+days)]
                signal_price = data.loc[signal_day, 'Close']

                # Calculate signal day's volume ratio
                avg_volume_20 = data['Volume'].loc[:signal_day].rolling(20).mean().iloc[-2]
                signal_volume = data.loc[signal_day, 'Volume']
                volume_ratio = signal_volume / avg_volume_20

                # Calculate signal day's RSI
                delta = data['Close'].loc[:signal_day].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                signal_rsi = rsi.iloc[-1]

                # Check if this would have been a signal
                if volume_ratio > 3 and 60 <= signal_rsi <= 80:

                    # Get the price N days after the signal
                    future_date = data.index[-(i)]
                    future_price = data.loc[future_date, 'Close']

                    # Calculate performance
                    performance = ((future_price - signal_price) / signal_price) * 100
                    lookback_prices.append({
                        'Signal_Date': signal_day.strftime('%Y-%m-%d'),
                        'Signal_Price': signal_price,
                        'Future_Date': future_date.strftime('%Y-%m-%d'),
                        'Future_Price': future_price,
                        'Performance_%': round(performance, 2),
                        'Profit': 'Yes' if performance > 0 else 'No'
                    })
            if lookback_prices:

                # Create a dataframe of past signals
                df = pd.DataFrame(lookback_prices)
                print("\nHistorical signals performance:")
                print(df.to_string(index=False))

                # Calculate win rate
                win_rate = (df['Performance_%'] > 0).mean() * 100
                avg_gain = df[df['Performance_%'] > 0]['Performance_%'].mean() if any(df['Performance_%'] > 0) else 0
                avg_loss = df[df['Performance_%'] < 0]['Performance_%'].mean() if any(df['Performance_%'] < 0) else 0
                print("\nSummary Statistics:")
                print(f"Win Rate: {win_rate:.2f}%")
                print(f"Average Gain: {avg_gain:.2f}%")
                print(f"Average Loss: {avg_loss:.2f}%")
                print(f"Risk-Reward Ratio: {abs(avg_gain/avg_loss):.2f}") if avg_loss != 0 else print("Risk-Reward Ratio: N/A")

                # Create visualization of backtest results
                self.visualize_backtest_results(df, symbol)
                return df
            else:
                print("No historical signals found for backtesting.")
                return None
        except Exception as e:
            logging.error(f"Error in backtesting for {symbol}: {e}")
            return None
    def visualize_backtest_results(self, backtest_df, symbol):
        """Create visualization of backtest results"""
        try:
            if backtest_df is None or len(backtest_df) == 0:
                print("No backtest data to visualize")
                return

            # Create charts directory if it doesn't exist
            charts_dir = 'charts'
            if os.path.isabs(charts_dir):
                os.makedirs(charts_dir, exist_ok=True)
            else:
                os.makedirs(os.path.join(os.getcwd(), charts_dir), exist_ok=True)

            # Create a figure for the backtest results
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})

            # Sort the dataframe by signal date
            backtest_df['Signal_Date'] = pd.to_datetime(backtest_df['Signal_Date'])
            backtest_df = backtest_df.sort_values('Signal_Date')

            # Performance plot
            ax1.plot(backtest_df['Signal_Date'], backtest_df['Performance_%'], 'o-', color='blue')
            ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax1.set_title(f'{symbol} - Historical Signal Performance', fontsize=15)
            ax1.set_ylabel('Performance (%)', fontsize=12)
            ax1.grid(True, alpha=0.3)

            # Color code the dots based on positive/negative performance
            for i, perf in enumerate(backtest_df['Performance_%']):
                color = 'green' if perf > 0 else 'red'
                ax1.plot(backtest_df['Signal_Date'].iloc[i], perf, 'o', color=color, markersize=8)

            # Distribution of performance
            ax2.hist(backtest_df['Performance_%'], bins=10, alpha=0.7, color='blue')
            ax2.axvline(x=0, color='r', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Performance (%)', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save the figure
            chart_path = os.path.join(charts_dir, f'{symbol.replace(".NS", "")}_backtest_results.png')
            plt.savefig(chart_path)
            plt.close()
            print(f"Backtest visualization saved to {chart_path}")
        except Exception as e:
            logging.error(f"Error creating backtest visualization: {e}")
    def save_report_to_csv(self, high_probability_stocks, medium_probability_stocks):
        """Save the report to CSV files for future reference"""
        try:

            # Create reports directory if it doesn't exist
            reports_dir = 'reports'
            if os.path.isabs(reports_dir):
                os.makedirs(reports_dir, exist_ok=True)
            else:
                os.makedirs(os.path.join(os.getcwd(), reports_dir), exist_ok=True)

            # Generate timestamp for filenames
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Save high probability stocks
            if high_probability_stocks:
                high_prob_df = pd.DataFrame(high_probability_stocks)
                high_prob_file = os.path.join(reports_dir, f'high_probability_stocks_{timestamp}.csv')
                high_prob_df.to_csv(high_prob_file, index=False)
                print(f"High probability stocks saved to {high_prob_file}")

            # Save medium probability stocks
            if medium_probability_stocks:
                medium_prob_df = pd.DataFrame(medium_probability_stocks)
                medium_prob_file = os.path.join(reports_dir, f'medium_probability_stocks_{timestamp}.csv')
                medium_prob_df.to_csv(medium_prob_file, index=False)
                print(f"Medium probability stocks saved to {medium_prob_file}")

            # Create a combined report
            all_stocks = high_probability_stocks + medium_probability_stocks
            if all_stocks:
                all_df = pd.DataFrame(all_stocks)
                all_file = os.path.join(reports_dir, f'all_momentum_stocks_{timestamp}.csv')
                all_df.to_csv(all_file, index=False)
                print(f"Combined report saved to {all_file}")
        except Exception as e:
            logging.error(f"Error saving report to CSV: {e}")
    def generate_early_warning_report(self):
        """Generate comprehensive early warning report"""
        logging.warning("🚨 EARLY WARNING SYSTEM REPORT")
        print("=" * 60)
        print(f"Analyzing {len(self.nse_stocks)} stocks for potential momentum...\n")
        high_probability_stocks = []
        medium_probability_stocks = []

        # Process stocks in batches to handle large lists
        total_stocks = len(self.nse_stocks)
        total_batches = (total_stocks + self.batch_size - 1) // self.batch_size
        for batch_num in range(total_batches):
            start_idx = batch_num * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_stocks)
            batch = self.nse_stocks[start_idx:end_idx]
            print(f"\nProcessing batch {batch_num+1}/{total_batches} ({len(batch)} stocks)...")
            print(f"Stocks {start_idx+1}-{end_idx} of {total_stocks}")
            for i, symbol in enumerate(batch):
                progress = (i + 1) / len(batch) * 100
                print(f"[{progress:.1f}%] Processing {symbol}...")
                try:
                    volume_signal = self.unusual_volume_detector(symbol)
                    momentum_signal = self.momentum_acceleration_detector(symbol)

                    # Both signals present - high chance of movement
                    if volume_signal and momentum_signal:
                        combined_score = 0
                        if volume_signal['signal_strength'] == 'HIGH':
                            combined_score += 3
                        elif volume_signal['signal_strength'] == 'MEDIUM':
                            combined_score += 2
                        combined_score += momentum_signal['momentum_score']
                        stock_data = {
                            'Symbol': symbol.replace('.NS', ''),
                            'Volume_Ratio': volume_signal['volume_ratio'],
                            'Price_Change_%': volume_signal['price_change'],
                            'RSI': momentum_signal['rsi'],
                            'MACD': momentum_signal['macd_signal'],
                            'Combined_Score': combined_score,
                            'Probability': 'HIGH' if combined_score >= 6 else 'MEDIUM'
                        }
                        if combined_score >= 5:
  # High probability threshold
                            high_probability_stocks.append(stock_data)
                        elif combined_score >= 3:
  # Medium probability
                            medium_probability_stocks.append(stock_data)

                    # Only volume signal - possible pump
                    elif volume_signal and not momentum_signal:
                        medium_probability_stocks.append({
                            'Symbol': symbol.replace('.NS', ''),
                            'Volume_Ratio': volume_signal['volume_ratio'],
                            'Price_Change_%': volume_signal['price_change'],
                            'RSI': 'N/A',
                            'MACD': 'N/A',
                            'Combined_Score': 2,
                            'Probability': 'MEDIUM (Volume Only)'
                        })
                except Exception as e:
                    logging.error(f"Error processing {symbol}: {e}")

                # Brief pause to avoid API rate limits
                time.sleep(0.5)

            # Add a pause between batches to avoid rate limits
            if batch_num < total_batches - 1:
                print(f"\nPausing for {self.timeout} seconds to avoid rate limits...")
                time.sleep(self.timeout)
        print("\n" + "=" * 60)

        # Generate charts for high probability stocks
        if high_probability_stocks:
            df = pd.DataFrame(high_probability_stocks)
            df = df.sort_values('Combined_Score', ascending=False)
            print("\n🎯 HIGH PROBABILITY CANDIDATES FOR THIS WEEK:")
            print(df.to_string(index=False))
            print("\nGenerating detailed charts for high probability stocks...")

            # Limit chart generation to top 10 stocks for large lists
            chart_limit = min(10, len(high_probability_stocks))
            for stock in high_probability_stocks[:chart_limit]:
                symbol = stock['Symbol'] + '.NS'
                self.analyze_and_plot_stock(symbol)

                # Sleep to avoid rate limiting
                time.sleep(1)

            # Perform backtesting on top high probability stocks
            print("\n📈 BACKTESTING HIGH PROBABILITY STOCKS")
            print("=" * 60)

            # Limit backtesting to top 5 stocks for large lists
            backtest_limit = min(5, len(high_probability_stocks))
            for stock in high_probability_stocks[:backtest_limit]:
                symbol = stock['Symbol'] + '.NS'
                print(f"\nBacktesting {symbol} to validate signal quality...")
                backtest_results = self.backtest_signal(symbol, days=5)

                # Sleep to avoid rate limiting
                time.sleep(1)
        else:
            print("\n❌ No high probability candidates found today.")
        if medium_probability_stocks:
            df = pd.DataFrame(medium_probability_stocks)
            df = df.sort_values('Combined_Score', ascending=False)
            print("\n📊 MEDIUM PROBABILITY CANDIDATES TO WATCH:")
            print(df.to_string(index=False))

            # Generate charts for top medium probability stocks (limit for large lists)
            if len(medium_probability_stocks) > 0:
                print("\nGenerating charts for top medium probability stocks...")
                chart_limit = min(5, len(medium_probability_stocks))
                for stock in medium_probability_stocks[:chart_limit]:
                    symbol = stock['Symbol'] + '.NS'
                    self.analyze_and_plot_stock(symbol)

                    # Sleep to avoid rate limiting
                    time.sleep(1)
        print(f"\n📅 Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("⚠️  This is probability-based analysis, not a guarantee!")

        # Save reports to CSV
        self.save_report_to_csv(high_probability_stocks, medium_probability_stocks)
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Early Warning System for Stock Momentum')

    # Input options
    parser.add_argument('-f', '--file', type=str, help='Path to file containing stock symbols (text or CSV)')
    parser.add_argument('-s', '--stocks', type=str, help='Comma-separated List[str] of stock symbols')

    # Batch processing options
    parser.add_argument('-b', '--batch-size', type=int, default=50,
                        help='Number of stocks to process in each batch (default: 50)')
    parser.add_argument('-t', '--timeout', type=int, default=10,
                        help='Timeout between batches in seconds (default: 10)')

    # Output options
    parser.add_argument('-o', '--output-dir', type=str, default='',
                        help='Directory to save output files (default: current directory)')
    return parser.parse_args()

# Usage
if __name__ == "__main__":

    # Parse command line arguments
    args = parse_arguments()

    # Set[str] output directory if specified
    if args.output_dir:
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            os.chdir(args.output_dir)
            print(f"Output will be saved to: {os.path.abspath(args.output_dir)}")
        except Exception as e:
            logging.error(f"Error setting output directory: {e}")

    # Create custom stock List[str] if provided via command line
    custom_stocks = None
    if args.stocks:
        custom_stocks = [s.strip() for s in args.stocks.split(',') if s.strip()]

        # Add .NS suffix if not present
        custom_stocks = [s if s.endswith('.NS') else f"{s}.NS" for s in custom_stocks]

    # Initialize and run the system
    ews = EarlyWarningSystem(
        custom_stocks=custom_stocks,
        input_file=args.file,
        batch_size=args.batch_size,
        timeout=args.timeout
    )
    ews.generate_early_warning_report()
