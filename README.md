# 📊 Daily Review Machine

**Intelligent Offline-First Trading Analysis System**

A comprehensive post-trading review machine built with Streamlit, Polars, scikit-learn, and LightGBM. Analyze your trades with ML-powered insights, regime detection, counterfactual analysis, and actionable recommendations.

---

## 🚀 Features

### 1. **Multi-Timeframe Review**
- Daily review (5-minute candles)
- Weekly review (5-minute candles)
- Monthly review (15/30-minute candles with toggle)
- Auto-resampling from source data

### 2. **GrowwAPI Integration**
- Real-time OHLCV data fetching
- Automatic caching for performance
- IST timezone support
- Trade log parsing and alignment

### 3. **Advanced Feature Engineering**
- VWAP distance and slope
- ATR and realized volatility
- Volume ratios (multi-period)
- Time-of-day encoding (sin/cos)
- Session phase detection
- All vectorized with Polars for speed

### 4. **Triple-Barrier Labeling**
- Meta-labeling for signal validation
- Realistic cost modeling (fees + slippage)
- Configurable TP/SL/timeout
- Time-series safe implementation

### 5. **ML Training Pipeline**
- Walk-forward validation with purge & embargo
- L2 Logistic Regression (primary)
- LightGBM with monotonic constraints
- Isotonic/Platt calibration
- Model snapshots with metadata

### 6. **Regime Detection**
- **Trend:** Strong directional movement
- **Chop:** Low volatility, range-bound
- **Breakout:** High volume + ATR expansion
- Per-regime performance metrics

### 7. **Counterfactual Engine**
- Entry timing shifts (-2 to +3 bars)
- Multiple exit strategies (TP/SL, time-based, ATR trail)
- TP/SL multiplier grid (0.75x to 1.25x)
- All scenarios include costs
- Best policy extraction with ΔR computation

### 8. **KPI Dashboard**
- Win rate, profit factor, avg R-multiple
- Per-regime breakdowns
- Exit reason analysis
- Avoidable loss and expected uplift
- Calibration and Brier scores

### 9. **Interactive What-If Lab**
- Visual trade replay with Plotly
- Live scenario simulation
- Full grid search across parameter space
- Side-by-side comparison

### 10. **Artifacts & Reports**
- Markdown summary reports
- CSV trade reviews with verdicts
- Interactive HTML charts
- JSONL logs for audit trail

---

## 📁 Project Structure

```
/app/
├── streamlit_app.py          # Main Streamlit application
├── cli.py                    # CLI automation tool
├── config/
│   ├── config.yaml           # Main configuration
│   └── costs.yaml            # Transaction costs
├── backend/
│   ├── core/
│   │   ├── data_ingestion.py       # GrowwAPI integration
│   │   ├── feature_engineering.py  # Vectorized features
│   │   ├── labeling.py             # Triple-barrier
│   │   ├── regime_detection.py     # Market regimes
│   │   ├── ml_trainer.py           # Walk-forward ML
│   │   ├── counterfactual.py       # What-if scenarios
│   │   └── kpi_computer.py         # Performance metrics
│   └── utils/
│       ├── data_loader.py          # Config & I/O
│       └── helpers.py              # Utility functions
├── data/
│   ├── bars/          # Cached OHLCV data
│   ├── trades/        # Trade logs
│   └── cache/         # API cache
├── models/            # ML model snapshots
├── reports/           # Generated reports
└── logs/              # Application logs
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- Groww API subscription & credentials

### Setup

```bash
# Install dependencies
pip install -r backend/requirements.txt

# Configure Groww credentials
# (Follow growwapi documentation for authentication setup)

# Run Streamlit app
streamlit run streamlit_app.py

# Or use CLI
python cli.py run --date 2025-01-15 --scope daily --symbol RELIANCE
```

---

## 📖 Usage

### Streamlit App (Recommended)

1. **Connect Groww API**
   - Click "Connect to Groww" in sidebar
   - Credentials auto-detected by growwapi

2. **Run Review**
   - Tab 1: Select scope (Daily/Weekly/Monthly)
   - Choose date and symbol
   - Click "Run Review"
   - View KPIs, charts, and regime breakdown

3. **Explore What-If Scenarios**
   - Tab 2: Select entry point on chart
   - Adjust parameters (entry shift, exit mode, TP/SL)
   - Run single simulation or full grid search

4. **Train Models**
   - Tab 3: Choose model type (Logistic/LightGBM)
   - Train on historical data
   - Load and compare snapshots

5. **Configure Regimes**
   - Tab 4: Adjust thresholds for trend/chop/breakout
   - View regime distribution pie chart
   - Save updated config

6. **Monitor Logs**
   - Tab 5: View application logs
   - Track errors and data issues

### CLI Automation

```bash
# Daily review
python cli.py run --date 2025-01-15 --scope daily --symbol RELIANCE

# Weekly review
python cli.py run --scope weekly --symbol INFY

# Monthly review with 15m candles
python cli.py run --scope monthly --symbol TCS

# Help
python cli.py run --help
```

**Note:** CLI mode uses synthetic data for demo. For real GrowwAPI data, use Streamlit app.

---

## ⚙️ Configuration

### `config/config.yaml`

```yaml
scopes:
  daily:
    candle_interval: 5
    horizon_bars: 12
    
counterfactual:
  entry_shifts: [-2, -1, 0, 1, 2, 3]
  exit_modes: ["tp_sl", "time_8", "atr_trail"]
  
regime:
  trend:
    min_vwap_slope: 0.0015
  chop:
    max_realized_vol: 0.025
  breakout:
    min_volume_ratio: 2.0
    
ml:
  models:
    logistic:
      C: 0.1
    lightgbm:
      max_depth: 4
```

### `config/costs.yaml`

```yaml
fees:
  equity:
    total: 6.3  # basis points
    
slippage:
  base_bps: 2.0
  impact_coef: 5.0
```

---

## 🎯 Key Concepts

### Meta-Labeling
Instead of predicting direction, models answer: "Should I take this base signal?" under realistic exit scenarios. This combines discretionary signals with ML filtering.

### Triple-Barrier Method
Each trade is simulated with three exit conditions:
1. Take-profit barrier (upside)
2. Stop-loss barrier (downside)
3. Time-based timeout

First touched wins. All include transaction costs.

### Regime-Aware Analysis
Market behavior varies by regime:
- **Trend:** Momentum strategies excel
- **Chop:** Mean-reversion works better
- **Breakout:** High risk/reward, timing critical

Per-regime KPIs reveal which setups to favor.

### Bounded Counterfactuals
Not "perfect hindsight" but realistic improvements:
- Entry shifts within reasonable range
- Exit strategies you could actually execute
- Costs included in every scenario
- Shows **achievable** uplift, not fantasy

---

## 📊 Sample Output

```
🚀 Daily Review for RELIANCE (2025-01-15)

📈 KPIs:
  Total Bars: 78
  Avg Volatility: 0.018
  Trend: 45%, Chop: 30%, Breakout: 15%
  
💼 Trades:
  Total: 7
  Win Rate: 57%
  Avg R: 0.8
  Profit Factor: 1.6
  
🔬 Counterfactual:
  Avoidable Loss: 0.8%
  Expected Uplift: 1.2%
  Best Policy: atr_trail with entry_shift=-1
  
✅ Top Recommendation:
  Focus on early-session breakouts in trend regime
  Tighten SL to 0.75x on choppy days
```

---

## 🧪 Testing

Sample trade log provided in `data/trades/sample_trades.csv`. Use for testing without real trades.

---

## 🤝 Contributing

This is a production-ready template. Customize for your strategy:

1. **Add custom features** in `feature_engineering.py`
2. **Tune regime logic** in `regime_detection.py`
3. **Extend counterfactual grid** in `counterfactual.py`
4. **Integrate other brokers** by wrapping their APIs in `data_ingestion.py`

---

## 📝 License

MIT License - Use freely, build something awesome!

---

## 🙏 Acknowledgments

- **Polars** for blazing-fast dataframes
- **Streamlit** for beautiful UI
- **scikit-learn & LightGBM** for ML power
- **Plotly** for interactive charts
- **GrowwAPI** for market data

---

**Built with intelligence. Trade with wisdom.** 🚀
