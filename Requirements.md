# System Requirements Specification – Upgraded Stock Analysis System

## 1. Core Functional Objectives

1. Detect **true unusual volume** using:

   * Volume z-score (statistical anomaly detection).
   * Rolling ratio vs. SMA(20).
   * Not just a fixed ×3 rule.
2. Use **multi-indicator confirmation** across RSI, MACD, ADX, ATR, and Volume Profile proxy.
3. Include **multi-timeframe confirmation** (daily + weekly).
4. Apply **market regime adaptation** (bull/bear/neutral) based on index breadth, moving averages, or ADX.
5. Produce a **composite probabilistic score (0–100)** with HIGH/MEDIUM/LOW buckets.
6. Integrate **robust backtesting** with:

   * Walk-forward validation.
   * Transaction costs + slippage.
   * No lookahead bias.
7. Implement **risk management** with:

   * ATR-based stops.
   * Volatility-scaled position sizing.
   * Max exposure limits.
8. (Optional) Train a **supervised ML model** using engineered features from signals.

---

## 2. Signal Requirements

### 2.1 Volume Anomalies

* **Daily Volume Ratio:** `V_today / SMA(V,20)`
* **Volume Z-Score:**
  `z = (V_today - mean(V,20)) / std(V,20)`
  Threshold: flag days with `z >= 2`
* **Both metrics required**:

  * Ratio detects scaled bursts.
  * Z-score captures statistical extremes.

### 2.2 Momentum

* **RSI(14):**

  * 60–70 → medium strength.
  * 70–85 → strong but caution.
  * 55–60 acceptable in weak signals.
* **MACD(12,26,9):**

  * MACD > Signal → positive score.
  * Positive histogram slope adds extra weight.
  * Normalized MACD momentum: `(MACD - Signal) / |Signal|`.

### 2.3 Trend Strength

* **ADX(14):**

  * ADX > 25 → strong trend.
  * ADX < 20 → weak/ignore.
* **Moving Averages:**

  * MA20 > MA50 = bullish confirmation.
  * Include slope check (optional refinement).

### 2.4 Volatility

* **ATR(14) relative to price.**
* High ATR → larger expected moves.
* ATR drives stop-loss distance and position sizing.

### 2.5 Relative Strength

* Compare stock returns vs. NIFTY or sector index.
* Rolling windows: 20-day and 50-day.
* Positive relative strength → score boost.

### 2.6 Volume Profile Proxy

* Approximate **Volume at Price** over last 90 days.
* Breakouts above high-volume nodes → bonus.

### 2.7 Market Regime

* Identify regime via:

  * NIFTY advance/decline ratio.
  * NIFTY 50 MA slope.
  * ADX of index.
* Thresholds adapt by regime:

  * Bull: looser confirmation (lower bar).
  * Bear: stricter confirmation (higher bar).

### 2.8 Liquidity & Market Cap

* Exclude stocks with:

  * Avg volume < defined threshold.
  * Market cap < defined threshold (e.g., ₹200cr).
* Exception: microcap plays if explicitly targeted.

### 2.9 Catalyst & Fundamental Checks (Optional)

* Exclude if within X days of earnings date.
* Screen for high-risk fundamentals:

  * Excessive debt.
  * Extremely high P/E.

---

## 3. Composite Scoring System

### 3.1 Component Weights

* Volume (ratio + z): **25**
* Momentum (RSI + MACD): **25**
* Trend Strength (MA + ADX): **15**
* Volatility & ATR: **10**
* Relative Strength: **10**
* Volume Profile Proxy: **10**
* Weekly Confirmation: +10 (bonus)

### 3.2 Volume Scoring

* z ≥ 3 → +15
* 2 ≤ z < 3 → +10
* Ratio ≥ 5 → +10
* Ratio ≥ 3 → +5

### 3.3 Momentum Scoring

* RSI 60–70 → +8
* RSI 70–80 → +12
* RSI ≥ 80 → +6 (caution penalty)
* MACD bullish crossover → +10
* MACD > 0 → +5

### 3.4 Trend Strength

* ADX > 25 → +8
* MA20 > MA50 → +7

### 3.5 Volatility

* ATR rising vs. 30-day avg → +6
* ATR absolute > threshold → +4

### 3.6 Relative Strength

* Outperforms sector over 20d by >X% → +10

### 3.7 Volume Profile

* Breakout above high-volume node → +10

### 3.8 Thresholds

* **HIGH:** Score ≥ 70
* **MEDIUM:** 45 ≤ Score < 70
* **LOW:** Score < 45

---

## 4. Multi-Timeframe Confirmation

* Weekly momentum confirms daily → +10 score bonus.
* Daily strong but weekly neutral → max Medium rating unless extra confirmations align.

---

## 5. Market-Adaptive Thresholds

* Compute NIFTY daily return volatility (30d).
* Adapt RSI sweet spot:

  * High volatility (>75th percentile) → expand to 62–82.
  * Low volatility → tighten to 58–78.

---

## 6. Backtesting Requirements

* **Methodology:**

  * Walk-forward validation: 2y train / 6m test rolling windows.
* **Metrics:**

  * Hit rate.
  * Avg return per trade.
  * Sharpe ratio (annualized).
  * Max drawdown.
  * Expectancy = (win% × avg\_win) − (loss% × avg\_loss).
* **Costs & Slippage:**

  * Transaction cost: \~0.05% per side.
  * Slippage: 0.1–0.5%.
  * NSE taxes/fees applied where relevant.
* **Validation:**

  * Bootstrap resampling of returns for p-value of positive expectancy.

---

## 7. Risk Management Rules

* **Stop Loss:** Entry − k × ATR (k = 1.5–2.5).
* **Position Sizing:**
  `risk_per_trade = portfolio_value × 0.01`
  `qty = floor(risk_per_trade / dollar_risk)`
* **Max Exposure:**

  * 10% of portfolio per position.
  * 30–50% total portfolio exposure cap.
* **Trailing Stops:**

  * Move stop to breakeven at 1.5× initial risk.
  * Then trail at 1.0 × ATR.
* **Max Daily New Positions:** configurable cap.

---

## 8. Machine Learning Extension (Optional)

* **Labeling:** Positive if forward return (5d) > X% or top-decile rank.
* **Features:** All indicators + volume profile + sector relative returns + fundamentals.
* **Models:** Gradient boosting (XGBoost/LightGBM), random forest, or neural nets.
* **Calibration:** Platt scaling or isotonic regression.
* **Explainability:** SHAP values.
* **Anti-leakage:** Features must only use data up to signal day.

---

## 9. Implementation Guidelines

* Precompute indicators on full history to reduce API load.
* Use **vectorized pandas/NumPy**; consider `numba` for speedups.
* Cache locally in feather/parquet formats.
* Respect rate limits; default batch size = 50, timeout = 5s.
* Monitor live performance with daily hit-rate report; adjust weights monthly.

---

## 10. Screener / Automation Rules

1. Universe filter:

   * avg\_volume\_20 > threshold (e.g., 100k).
   * price > ₹10.
   * market\_cap > ₹200cr.
2. Volume surge:

   * volume\_today > 3 × SMA(volume,20) OR zscore ≥ 2.
3. Momentum:

   * RSI(14) between 60–80.
   * MACD > signal.
4. Pass candidates to composite scoring function.

---

## 11. Deliverables

* Python module:

  * Indicator engine.
  * Composite scoring system.
  * Multi-timeframe confirmation logic.
* Backtesting harness:

  * Walk-forward validation.
  * Performance reporting.
* (Optional) ML pipeline:

  * Feature engineering.
  * Training/evaluation.
  * SHAP explainability.
* Screener queries for Chartink/TradingView/Screener.in.
* NSE universe run example:

  * CSV outputs of HIGH/MEDIUM/LOW.
  * Charts for top N signals.