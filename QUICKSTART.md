# ⚡ Quick Start Guide - Daily Review Machine

Get up and running in 5 minutes!

---

## 🎯 What You'll Build

A sophisticated trading review system that:
- ✅ Fetches real market data via GrowwAPI
- ✅ Computes 11+ technical features (VWAP, ATR, volume ratios)
- ✅ Detects market regimes (Trend/Chop/Breakout)
- ✅ Trains calibrated ML models (Logistic + LightGBM)
- ✅ Runs counterfactual "what-if" analysis
- ✅ Generates actionable reports

---

## 📦 Installation

### Prerequisites
- Python 3.11+
- Groww API subscription

### Install Dependencies

```bash
cd /app
pip install -r backend/requirements.txt
```

**Installed packages:**
- `streamlit` - Interactive web UI
- `polars` - Blazing fast dataframes
- `scikit-learn` - ML models
- `lightgbm` - Gradient boosting
- `plotly` - Interactive charts
- `growwapi` - Market data integration
- `pyyaml` - Configuration

---

## 🚀 Launch Options

### Option 1: Streamlit App (Recommended)

**Best for:** Interactive analysis, model training, visual exploration

```bash
streamlit run streamlit_app.py
```

Then open: http://localhost:8502

**Features:**
- 5 interactive tabs
- Real-time data fetching
- Visual what-if scenarios
- Model training UI
- Regime threshold controls

### Option 2: CLI Automation

**Best for:** Scheduled reviews, batch processing, scripting

```bash
# Daily review
python cli.py run --date 2025-01-15 --scope daily --symbol RELIANCE

# Weekly review
python cli.py run --scope weekly --symbol INFY

# Monthly review (15m candles)
python cli.py run --scope monthly --symbol TCS
```

**Output:**
- `reports/YYYY-MM-DD/summary.md` - Markdown report
- `reports/YYYY-MM-DD/trades_review.csv` - Trade-by-trade analysis

---

## 🎬 First Run Tutorial

### Step 1: Configure Groww API

The `growwapi` library automatically detects your credentials. Make sure you have:

1. Active Groww subscription
2. API credentials configured per [GrowwAPI docs](https://github.com/Groww-OSS/groww-api-python)

### Step 2: Launch Streamlit

```bash
streamlit run streamlit_app.py
```

### Step 3: Connect API

1. Click **"Connect to Groww"** in sidebar
2. Wait for green "Connected" status

### Step 4: Run Your First Review

**Tab 1: Review Day/Week/Month**

1. Select **Review Scope:** Daily (5m)
2. Choose **Date:** Today
3. Enter **Symbol:** RELIANCE
4. Click **🚀 Run Review**

You'll see:
- 📊 KPI tiles (total bars, volatility, regime distribution)
- 📉 Interactive price chart with VWAP
- 🎨 Volume bars colored by regime

### Step 5: Explore What-If Scenarios

**Tab 2: What-If Lab**

1. Select loaded symbol (RELIANCE)
2. Use slider to choose entry point
3. Adjust parameters:
   - Entry shift: -2 to +3 bars
   - Exit mode: tp_sl, time_8, atr_trail
   - TP/SL multipliers
4. Click **🔄 Simulate** for single scenario
5. Or **🚀 Run Full Grid** to explore all combinations

**Insight:** See how timing and exit strategy affect PnL

### Step 6: Train ML Model (Optional)

**Tab 3: Model Manager**

1. Choose **Model Type:** logistic (faster) or lightgbm (more accurate)
2. Set **Snapshot Date:** e.g., 20250115
3. Click **🚀 Train Model**

Model will:
- Create meta-labels via triple-barrier
- Walk-forward validation
- Calibrate probabilities
- Save snapshot to `models/`

**Metrics shown:**
- AUC (discrimination)
- Brier score (calibration)
- Log loss

### Step 7: Configure Regimes

**Tab 4: Regime Controls**

Adjust thresholds to match your market:

- **Trend:** Increase `min_vwap_slope` for stronger trends only
- **Chop:** Lower `max_realized_vol` to tighten chop definition
- **Breakout:** Raise `min_volume_ratio` for cleaner breakouts

Click **💾 Save Configuration** to persist changes.

---

## 📁 Project Structure Quick Reference

```
/app/
├── streamlit_app.py       # Main UI
├── cli.py                 # CLI tool
├── config/
│   ├── config.yaml        # Review & ML settings
│   └── costs.yaml         # Fees & slippage
├── backend/core/          # Core modules
│   ├── data_ingestion.py     # GrowwAPI
│   ├── feature_engineering.py # Features
│   ├── labeling.py           # Triple-barrier
│   ├── regime_detection.py   # Regimes
│   ├── ml_trainer.py         # ML training
│   ├── counterfactual.py     # What-if engine
│   └── kpi_computer.py       # Metrics
├── data/
│   ├── bars/          # Cached OHLCV
│   ├── trades/        # Trade logs
│   └── cache/         # API cache
├── models/            # Trained models
├── reports/           # Generated reports
└── logs/              # Application logs
```

---

## 🎯 Common Use Cases

### Use Case 1: End-of-Day Review

**Goal:** Analyze today's trades after market close

```bash
# CLI approach (automatable)
python cli.py run --date $(date +%Y-%m-%d) --scope daily --symbol NIFTY50

# Streamlit approach (visual)
streamlit run streamlit_app.py
# Then: Tab 1 → Select today → Run Review
```

**Output:** KPIs, regime breakdown, counterfactual best policies

### Use Case 2: Backtest Strategy Changes

**Goal:** "What if I used tighter stops last week?"

```bash
# In Streamlit: Tab 2 (What-If Lab)
# 1. Load weekly data
# 2. For each trade entry, run grid with sl_mult=0.75
# 3. Compare ΔR vs actual
```

### Use Case 3: Regime-Based Rules

**Goal:** Optimize strategy per regime

```bash
# Train model on historical data
# Tab 3: Train model with regime features
# Tab 4: View regime performance
# Insight: "Only trade breakouts in low-chop regimes"
```

### Use Case 4: Scheduled Automation

**Goal:** Daily reports emailed at 4 PM

```bash
# Add to crontab
0 16 * * 1-5 cd /app && python cli.py run --scope daily | mail -s "Daily Review" trader@example.com
```

---

## 🐛 Troubleshooting

### "Groww API Not Connected"

**Fix:**
```bash
# Verify credentials
python -c "import growwapi; print(growwapi.Groww())"

# Check environment
echo $GROWW_API_KEY
```

### "No data returned"

**Causes:**
1. Market closed (only fetch during 9:15 AM - 3:30 PM IST)
2. Invalid symbol name
3. Date too far in past (API limits)

**Fix:** Use recent trading day and correct symbol format

### "Insufficient data for features"

**Cause:** Less than 20 bars fetched

**Fix:** Expand date range or reduce `atr_period` in config

### Streamlit caching issues

**Fix:**
```bash
# Clear cache
rm -rf ~/.streamlit/cache
rm -rf data/cache/*

# Restart app
pkill -f streamlit && streamlit run streamlit_app.py
```

---

## 📚 Learn More

### Configuration Files

**config/config.yaml:**
- `scopes` - Time intervals per review type
- `counterfactual` - What-if grid parameters
- `regime` - Threshold definitions
- `ml` - Model hyperparameters

**config/costs.yaml:**
- `fees.equity.total` - Total transaction cost (bps)
- `slippage.base_bps` - Base slippage

**Edit and restart app to apply changes**

### Sample Data

Included: `data/trades/sample_trades.csv`

Format:
```csv
ts_open,ts_close,symbol,side,qty,entry_px,exit_px,tp,sl,base_score,reason,notes
2025-01-15 10:30:00,2025-01-15 11:45:00,RELIANCE,long,100,2500.00,2537.50,2537.50,2481.25,0.75,tp,Strong momentum
```

Load via Streamlit or place in `data/trades/` for batch analysis.

---

## 🚀 Next Steps

1. ✅ Run first daily review
2. ✅ Explore what-if scenarios
3. ✅ Train your first model
4. ✅ Set up cron for automation
5. 📖 Read [README.md](README.md) for deep dive
6. 🚢 See [DEPLOYMENT.md](DEPLOYMENT.md) for production setup

---

## 💡 Tips for Success

1. **Start with Daily Reviews**
   - Smaller data, faster iteration
   - Build intuition before weekly/monthly

2. **Use Mock Mode for Testing**
   - CLI uses synthetic data when Groww not connected
   - Perfect for learning the system

3. **Cache Aggressively**
   - GrowwAPI data is cached automatically
   - Speeds up repeated analysis

4. **Compare Regimes**
   - Tab 4 shows per-regime performance
   - Adapt strategy to market conditions

5. **Track Model Drift**
   - Retrain weekly with fresh data
   - Compare AUC trends over time

---

## 🎓 Concepts to Understand

### Meta-Labeling
Not predicting direction, but filtering signals:
- Input: Base signal (e.g., MA crossover, scanner score)
- Output: "Take this signal?" (1) or "Skip" (0)
- Advantage: Combines discretionary + ML

### Triple-Barrier
Realistic exit simulation:
1. Take-profit hit → win
2. Stop-loss hit → loss
3. Timeout → whatever close price
All include transaction costs.

### Regime Detection
Market behavior varies:
- **Trend:** Momentum works
- **Chop:** Mean-reversion better
- **Breakout:** High risk/reward, timing critical

Optimize per-regime for better results.

### Counterfactuals
Not hindsight, but **realistic improvements**:
- Entry shifts within ±3 bars (achievable)
- Exit strategies you can actually code
- All costs included
Shows **avoidable loss**, not fantasy gains.

---

## ✨ You're Ready!

Your trading review machine is set up and ready. Start with a simple daily review, explore the what-if lab, and gradually build your intuition.

**Happy trading!** 📈

---

*Questions? Check logs/app.log or logs/cli.log for detailed traces.*
