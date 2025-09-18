# NSE Stock Screener

A Python tool to fetch, validate, and screen NSE-listed stocks for high-momentum setups.  
Designed to help uncover under-followed stocks with potential to “bloom.”

---

## 🚀 What It Does

- Fetches stock symbols from NSE (beyond just the big indices).  
- Validates tickers using Yahoo Finance.  
- Saves outputs as text/CSV for downstream analysis.  
- Built to be extended with technical or fundamental filters.

---

## 🔍 Why This Exists

The Nifty 500 and other large-cap indices are well-known; many tools focus only there. The opportunity lies in digging deeper — small or mid-cap stocks, or less obvious ones — to find higher-growth potential. This tool gives you a foundation: data + validation + structure.

---

## 📂 Project Structure

```

nse-stock-screener/
│
├── src/                   # Python source scripts
│     └── fetch\_symbols.py  # Main symbol-fetching logic
│
├── data/                  # Outputs and temporary data
│     ├── symbols.txt       # Saved symbols
│     └── temp/             # Temporary files during fetch
│
├── .gitignore             # Which files/folders to ignore
├── requirements.txt       # Python dependencies
└── README.md              # This file

````

---

## 🛠 Setup & Usage

1. Clone the repo:
   ```bash
   git clone https://github.com/sandeep-chauhan-self/nse-stock-screener.git
   cd nse-stock-screener
    ```

2. (Optional but best) Create a virtual environment and activate it:

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install requirements:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the main script:

   ```bash
   python src/fetch_symbols.py
   ```

5. The symbols will be saved in `data/symbols.txt`.
   Temporary files (if any) go into `data/temp/`.

---

## ✔️ Validation

* The script picks a small random subset of fetched symbols and checks via Yahoo Finance whether they appear valid.
* Helps filter out delisted or incorrect tickers.

---

## ⚙️ Possible Next Steps / Improvements

* Incorporate filters (volume spikes, PE/PB, promoter holding, etc.).
* Add functionality for technical indicators (RSI, MACD, moving averages).
* Automate discovering “rising stars”: small-cap stocks rising steadily.
* Build UI or dashboard for visual exploration.
* Backtest signals.

---

## 🪃 Dependencies

* Python 3.x
* pandas
* requests
* yfinance

(Check `requirements.txt` for exact versions.)

---

## 📄 License & Disclaimer

This repo is provided as-is, for educational or personal use. It **does not** provide investment advice. Always verify data and do your own due diligence before making investment decisions.

---

## 📬 Feedback & Contribution

Ideas, bug reports, or pull requests are welcome. If you've built an interesting filter or module, feel free to contribute!
