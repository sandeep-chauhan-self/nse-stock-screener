# NSE Stock Screener

A comprehensive **Python-based stock screener** for the **NSE (National Stock Exchange of India)**.  
This tool is designed to detect unusual trading activity, validate stock tickers, and generate actionable datasets for deeper analysis.

---

## 🚀 Features

- Fetches NSE-listed stock symbols (beyond Nifty indices).
- Validates tickers using **Yahoo Finance API (yfinance)**.
- Detects **unusual trading volume** using:
  - Z-Score based anomaly detection
  - Rolling volume ratio comparison
- Supports **multi-indicator confirmation** across:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - ADX (Average Directional Index)
  - ATR (Average True Range)
  - Volume Profile
- Saves outputs in **CSV/Text format** for downstream analysis.
- Modular design: extendable for **fundamental or technical filters**.
- Backtesting and charting support (CSV-based results).

---

## 📂 Project Structure

```
nse-stock-screener/
│
├── src/                     # Core Python scripts
│   ├── fetch_symbols.py      # Fetch NSE symbols
│   ├── indicators_engine.py  # Compute technical indicators
│   ├── backtest.py           # Backtesting logic
│   └── utils.py              # Helper functions
│
├── scripts/                  # Automation scripts
│   └── start.bat             # Windows launcher
│
├── data/                     # Output & temporary files
│   ├── symbols.txt
│   ├── results/              # Indicator + backtest results
│   └── temp/
│
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignored files
└── README.md                 # Documentation
```

---

## ⚡ Installation & Usage

1. Clone the repository:

```bash
git clone https://github.com/sandeep-chauhan-self/nse-stock-screener.git
cd nse-stock-screener
```

2. (Recommended) Create a virtual environment:

```bash
python -m venv venv
# On Linux/Mac
source venv/bin/activate
# On Windows
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the main script:

```bash
python src/fetch_symbols.py
```

5. Analyze a single stock with the full indicator suite:

```bash
python src/indicators_engine.py --symbol RELIANCE.NS
```

6. Results are saved as CSV inside the `data/results/` folder.

---

## 📊 Example Output

| Symbol      | RSI  | MACD Signal | ADX | ATR  | Volume Z-Score | Action Signal |
|-------------|------|-------------|-----|------|----------------|---------------|
| RELIANCE.NS | 62   | Bullish     | 28  | 15.3 | +2.1           | ✅ Buy Watch  |
| TCS.NS      | 41   | Bearish     | 19  | 10.8 | -0.7           | ❌ Avoid      |

---

## 🔮 Roadmap

- [ ] Add more **fundamental filters** (PE, PB, ROE, promoter holdings).
- [ ] Automate **multi-day signal scanning**.
- [ ] Integrate **plotting/charting** for visual insights.
- [ ] Deploy as a **Flask/Django dashboard**.
- [ ] Add **machine learning models** for predictive screening.

---

## ⚙️ Dependencies

- Python 3.8+
- pandas
- numpy
- requests
- yfinance
- matplotlib (for charts)
- ta (technical analysis library)

Check `requirements.txt` for exact versions.

---

## 📜 License & Disclaimer

This project is open-source and provided **for educational & research purposes only**.  
It does **not constitute financial advice**. Use at your own risk.

---

## 🤝 Contributions

Contributions are welcome!  
- Fork the repo, implement improvements, and open a pull request.  
- Report bugs or suggest features via [GitHub Issues](../../issues).  

---

## 👨‍💻 Author

Created and maintained by **Sandeep Chauhan**  
📌 Focused on financial technology & trading system development.

