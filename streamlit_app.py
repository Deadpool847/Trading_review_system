"""Daily Review Machine - Streamlit Application"""
import streamlit as st
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import logging
import yaml
import json

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / 'backend'))

# Import core modules
from core.data_ingestion import DataIngestion
from core.feature_engineering import FeatureEngineer
from core.labeling import TripleBarrierLabeler
from core.regime_detection import RegimeDetector
from core.ml_trainer import MLTrainer
from core.counterfactual import CounterfactualEngine
from core.kpi_computer import KPIComputer
from core.stock_analyzer import StockAnalyzer
from utils.data_loader import load_config, load_costs_config, save_trades_review, save_summary_report
from utils.helpers import get_ist_timezone, format_date_for_api

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure directories exist
for dir_path in ['data/bars', 'data/trades', 'data/cache', 'models', 'reports', 'logs']:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

# Page config
st.set_page_config(
    page_title="Daily Review Machine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border-left: 4px solid #6366f1;
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }
    
    .positive {
        color: #10b981;
    }
    
    .negative {
        color: #ef4444;
    }
    
    .tab-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(99, 102, 241, 0.3);
    }
    
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'groww_client' not in st.session_state:
    st.session_state.groww_client = None
if 'config' not in st.session_state:
    st.session_state.config = load_config()
if 'costs_config' not in st.session_state:
    st.session_state.costs_config = load_costs_config()
if 'cached_bars' not in st.session_state:
    st.session_state.cached_bars = {}
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'trade_logs' not in st.session_state:
    st.session_state.trade_logs = {'scanner': [], 'entries': []}
if 'analysis_cache' not in st.session_state:
    st.session_state.analysis_cache = {}

# Sidebar - Groww Authentication
st.sidebar.title("🔐 Groww Authentication")
st.sidebar.markdown("---")

# Groww auth section
with st.sidebar.expander("Configure Groww API", expanded=True):
    st.info("Your Groww API credentials will be automatically detected by growwapi library.")
    
    try:
        import growwapi
        
        if st.button("🔗 Connect to Groww", key="connect_groww"):
            with st.spinner("Connecting to Groww..."):
                try:
                    # Initialize Groww client (will read credentials from environment/config)
                    groww_client = growwapi.Groww()
                    st.session_state.groww_client = groww_client
                    st.success("✅ Connected to Groww API!")
                except Exception as e:
                    st.error(f"❌ Connection failed: {e}")
                    st.info("Make sure your Groww credentials are properly configured.")
    except ImportError:
        st.error("growwapi library not found. Please install it.")

if st.session_state.groww_client:
    st.sidebar.success("🟢 Groww API Connected")
else:
    st.sidebar.warning("🔴 Groww API Not Connected")

st.sidebar.markdown("---")

# Main App
st.title("📊 Daily Review Machine")
st.markdown("### *Intelligent Post-Trading Analysis System*")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Review Day/Week/Month",
    "🔬 What-If Lab",
    "🤖 Model Manager",
    "🎯 Regime Controls",
    "📋 Logs"
])

# =======================
# TAB 1: Stock Analysis (Standalone)
# =======================
with tab1:
    st.markdown('<div class="tab-header"><h2>📊 Stock Analysis Brain</h2><p>Comprehensive stock analysis with actionable insights</p></div>', unsafe_allow_html=True)
    
    # Custom timeframe selection
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        symbol_input = st.text_input(
            "Stock Symbol",
            "NSE-RELIANCE",
            help="Format: NSE-SYMBOL (e.g., NSE-RELIANCE, NSE-INFY)",
            key="symbol_input"
        )
    
    with col2:
        interval_options = {
            "1 min": 1,
            "2 min": 2,
            "3 min": 3,
            "5 min": 5,
            "10 min": 10,
            "15 min": 15,
            "30 min": 30,
            "1 hour": 60,
            "4 hours": 240,
            "1 day": 1440
        }
        interval_choice = st.selectbox(
            "Candle Interval",
            list(interval_options.keys()),
            index=3,  # Default to 5 min
            key="interval_choice"
        )
        interval_minutes = interval_options[interval_choice]
    
    with col3:
        # Duration limits based on interval
        duration_limits = {
            1: 30, 2: 30, 3: 30, 5: 30,
            10: 90, 15: 90, 30: 90,
            60: 180, 240: 180, 1440: 180
        }
        max_days = duration_limits.get(interval_minutes, 30)
        
        days_back = st.number_input(
            f"Days Back (max {max_days})",
            min_value=1,
            max_value=max_days,
            value=min(7, max_days),
            key="days_back"
        )
    
    with col4:
        # Optional trade log upload
        log_file = st.file_uploader(
            "Trade Logs (Optional)",
            type=['log', 'txt'],
            help="Upload your trade logs to merge with analysis",
            key="log_file"
        )
    
    # Info about duration limits
    if interval_minutes <= 5:
        st.info("ℹ️ 1-5 min candles: Max 30 days | 10-30 min: Max 90 days | 1hr+: Max 180 days")
    elif interval_minutes <= 30:
        st.info("ℹ️ 10-30 min candles: Max 90 days | 1-5 min: Max 30 days | 1hr+: Max 180 days")
    else:
        st.info("ℹ️ 1hr+ candles: Max 180 days | 10-30 min: Max 90 days | 1-5 min: Max 30 days")
    
    if st.button("🔍 Analyze Stock", key="run_analysis", use_container_width=True):
        with st.spinner("🧠 Running comprehensive analysis..."):
            try:
                # Calculate time range
                end_time = datetime.now()
                start_time = end_time - timedelta(days=days_back)
                
                # Ensure market hours for intraday
                if interval_minutes < 1440:  # Not daily
                    start_time = start_time.replace(hour=9, minute=15, second=0)
                    end_time = end_time.replace(hour=15, minute=30, second=0)
                
                start_time_str = format_date_for_api(start_time)
                end_time_str = format_date_for_api(end_time)
                
                # Process trade logs if uploaded
                scanner_results = []
                trade_entries = []
                
                if log_file is not None:
                    # Save uploaded file temporarily
                    temp_log_path = Path("data/trades/temp_upload.log")
                    temp_log_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(temp_log_path, 'wb') as f:
                        f.write(log_file.getvalue())
                    
                    ingestion = DataIngestion()
                    scanner_results, trade_entries = ingestion.load_trade_logs(str(temp_log_path))
                    
                    if scanner_results or trade_entries:
                        st.success(f"✅ Loaded {len(scanner_results)} scanner results, {len(trade_entries)} trade entries")
                        st.session_state.trade_logs = {'scanner': scanner_results, 'entries': trade_entries}
                
                # Fetch data (use mock if no Groww connection)
                ingestion = DataIngestion()
                
                if st.session_state.groww_client:
                    bars_df = ingestion.fetch_ohlcv_bars(
                        st.session_state.groww_client,
                        symbol_input,
                        start_time_str,
                        end_time_str,
                        interval_minutes
                    )
                else:
                    # Generate mock data for standalone analysis
                    st.warning("⚠️ No Groww connection - using synthetic data for demo")
                    
                    from datetime import timezone as tz
                    n_bars = min(int((days_back * 6.25 * 60) / interval_minutes), 500)  # 6.25 hours per day
                    timestamps = [start_time + timedelta(minutes=i*interval_minutes) for i in range(n_bars)]
                    base_price = 2500.0
                    
                    bars_data = []
                    for i, ts in enumerate(timestamps):
                        price_change = np.random.randn() * 10
                        close = base_price + price_change
                        high = close + abs(np.random.randn() * 5)
                        low = close - abs(np.random.randn() * 5)
                        open_price = bars_data[-1]['c'] if bars_data else close
                        volume = int(10000 + np.random.randn() * 2000)
                        
                        bars_data.append({
                            'ts': ts.replace(tzinfo=tz.utc),
                            'symbol': symbol_input,
                            'o': open_price,
                            'h': high,
                            'l': low,
                            'c': close,
                            'v': volume,
                            'vwap': (high + low + close) / 3
                        })
                        base_price = close
                    
                    bars_df = pl.DataFrame(bars_data)
                    
                if len(bars_df) == 0:
                    st.error("❌ No data available")
                else:
                    # Feature engineering
                    engineer = FeatureEngineer(st.session_state.config)
                    bars_df = engineer.compute_features(bars_df)
                    
                    # Regime detection
                    regime_detector = RegimeDetector(st.session_state.config)
                    bars_df = regime_detector.detect_regime(bars_df)
                    
                    # Advanced stock analysis
                    analyzer = StockAnalyzer(st.session_state.config)
                    price_analysis = analyzer.analyze_price_action(bars_df)
                    
                    # Get dominant regime
                    regime_stats = regime_detector.get_regime_stats(bars_df)
                    dominant_regime = max(regime_stats.items(), key=lambda x: x[1]['count'])[0] if regime_stats else 'unknown'
                    
                    # Generate insights
                    insights = analyzer.generate_insights(bars_df, price_analysis, dominant_regime)
                    
                    # Merge with trade logs if available
                    markers = analyzer.merge_with_trade_logs(
                        bars_df,
                        st.session_state.trade_logs['scanner'],
                        st.session_state.trade_logs['entries'],
                        symbol_input
                    )
                    
                    # Generate summary
                    summary_text = analyzer.generate_summary(
                        symbol_input,
                        bars_df,
                        price_analysis,
                        insights,
                        dominant_regime
                    )
                    
                    # Cache for other tabs
                    st.session_state.cached_bars[symbol_input] = bars_df
                    st.session_state.analysis_cache[symbol_input] = {
                        'analysis': price_analysis,
                        'insights': insights,
                        'regime': dominant_regime,
                        'markers': markers,
                        'summary': summary_text
                    }
                    
                    # Display comprehensive analysis
                    st.markdown("---")
                    st.markdown("### 📊 Market Overview")
                        
                        metric_cols = st.columns(5)
                        
                        with metric_cols[0]:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{len(bars_df)}</div>
                                <div class="metric-label">Total Bars</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with metric_cols[1]:
                            avg_vol = bars_df['realized_vol'].mean() if 'realized_vol' in bars_df.columns else 0
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{avg_vol:.3f}</div>
                                <div class="metric-label">Avg Volatility</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with metric_cols[2]:
                            regime_stats = regime_detector.get_regime_stats(bars_df)
                            trend_pct = regime_stats.get('trend', {}).get('pct', 0)
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{trend_pct:.1f}%</div>
                                <div class="metric-label">Trend Bars</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with metric_cols[3]:
                            chop_pct = regime_stats.get('chop', {}).get('pct', 0)
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{chop_pct:.1f}%</div>
                                <div class="metric-label">Chop Bars</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with metric_cols[4]:
                            breakout_pct = regime_stats.get('breakout', {}).get('pct', 0)
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{breakout_pct:.1f}%</div>
                                <div class="metric-label">Breakout Bars</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Chart
                        st.markdown("### 📉 Price Action & Regime")
                        
                        fig = make_subplots(
                            rows=2, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.05,
                            row_heights=[0.7, 0.3],
                            subplot_titles=("Price & VWAP", "Volume & Regime")
                        )
                        
                        # Candlestick
                        bars_pd = bars_df.to_pandas()
                        fig.add_trace(
                            go.Candlestick(
                                x=bars_pd['ts'],
                                open=bars_pd['o'],
                                high=bars_pd['h'],
                                low=bars_pd['l'],
                                close=bars_pd['c'],
                                name="Price"
                            ),
                            row=1, col=1
                        )
                        
                        # VWAP
                        fig.add_trace(
                            go.Scatter(
                                x=bars_pd['ts'],
                                y=bars_pd['vwap'],
                                name="VWAP",
                                line=dict(color='#f59e0b', width=2)
                            ),
                            row=1, col=1
                        )
                        
                        # Volume bars colored by regime
                        regime_colors = {
                            'trend': '#10b981',
                            'chop': '#64748b',
                            'breakout': '#f59e0b',
                            'unknown': '#94a3b8'
                        }
                        
                        for regime, color in regime_colors.items():
                            regime_bars = bars_pd[bars_pd['regime'] == regime]
                            if len(regime_bars) > 0:
                                fig.add_trace(
                                    go.Bar(
                                        x=regime_bars['ts'],
                                        y=regime_bars['v'],
                                        name=regime.capitalize(),
                                        marker_color=color,
                                        showlegend=True
                                    ),
                                    row=2, col=1
                                )
                        
                        fig.update_layout(
                            height=700,
                            template='plotly_white',
                            xaxis_rangeslider_visible=False,
                            hovermode='x unified',
                            font=dict(family='Inter, sans-serif')
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Cache for other tabs
                        st.session_state.cached_bars[symbol_input] = bars_df
                        st.success(f"✅ Review complete! {len(bars_df)} bars analyzed.")
                        
                except Exception as e:
                    st.error(f"❌ Error during review: {e}")
                    logger.error(f"Review error: {e}", exc_info=True)

# =======================
# TAB 2: What-If Lab
# =======================
with tab2:
    st.markdown('<div class="tab-header"><h2>🔬 What-If Laboratory</h2><p>Explore counterfactual scenarios</p></div>', unsafe_allow_html=True)
    
    if not st.session_state.cached_bars:
        st.info("👆 Please run a review first to load data")
    else:
        symbol = st.selectbox("Select Symbol", list(st.session_state.cached_bars.keys()), key="whatif_symbol")
        bars_df = st.session_state.cached_bars[symbol]
        
        # Trade selector
        entry_idx = st.slider(
            "Select Entry Bar Index",
            0,
            len(bars_df) - 10,
            len(bars_df) // 2,
            key="entry_idx"
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📊 Trade Visualization")
            
            # Get window around entry
            window_start = max(0, entry_idx - 20)
            window_end = min(len(bars_df), entry_idx + 30)
            window_bars = bars_df[window_start:window_end]
            
            bars_pd = window_bars.to_pandas()
            
            fig = go.Figure()
            
            # Candlestick
            fig.add_trace(go.Candlestick(
                x=bars_pd['ts'],
                open=bars_pd['o'],
                high=bars_pd['h'],
                low=bars_pd['l'],
                close=bars_pd['c'],
                name="Price"
            ))
            
            # VWAP
            fig.add_trace(go.Scatter(
                x=bars_pd['ts'],
                y=bars_pd['vwap'],
                name="VWAP",
                line=dict(color='#f59e0b', width=2)
            ))
            
            # Entry point
            entry_price = bars_df[entry_idx, 'c']
            entry_time = bars_df[entry_idx, 'ts']
            
            fig.add_trace(go.Scatter(
                x=[entry_time],
                y=[entry_price],
                mode='markers',
                marker=dict(size=15, color='#10b981', symbol='triangle-up'),
                name='Entry',
                showlegend=True
            ))
            
            fig.update_layout(
                height=500,
                template='plotly_white',
                xaxis_rangeslider_visible=False,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### ⚙️ Scenario Controls")
            
            entry_shift = st.select_slider(
                "Entry Shift (bars)",
                options=[-3, -2, -1, 0, 1, 2, 3],
                value=0
            )
            
            exit_mode = st.selectbox(
                "Exit Strategy",
                ["tp_sl", "time_8", "time_12", "atr_trail"]
            )
            
            tp_mult = st.slider("TP Multiplier", 0.5, 2.0, 1.0, 0.25)
            sl_mult = st.slider("SL Multiplier", 0.5, 2.0, 1.0, 0.25)
            
            if st.button("🔄 Simulate", use_container_width=True):
                cf_engine = CounterfactualEngine(st.session_state.config, st.session_state.costs_config)
                
                result = cf_engine.simulate_trade(
                    bars_df,
                    entry_idx,
                    entry_shift,
                    exit_mode,
                    tp_mult,
                    sl_mult
                )
                
                st.markdown("#### Results")
                
                pnl_class = "positive" if result['pnl_pct'] > 0 else "negative"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value {pnl_class}">{result['pnl_pct']:.2%}</div>
                    <div class="metric-label">PnL</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.write(f"**Exit:** {result['exit_reason']}")
                st.write(f"**Bars Held:** {result['bars_held']}")
                st.write(f"**Entry Price:** ₹{result['entry_price']:.2f}")
                st.write(f"**Exit Price:** ₹{result['exit_price']:.2f}")
        
        # Full grid search
        st.markdown("---")
        if st.button("🚀 Run Full Counterfactual Grid", use_container_width=True):
            with st.spinner("Running grid search..."):
                cf_engine = CounterfactualEngine(st.session_state.config, st.session_state.costs_config)
                results = cf_engine.run_grid_search(bars_df, entry_idx)
                
                st.markdown(f"### 📊 Top 10 Scenarios (out of {len(results)})")
                
                top_results = results[:10]
                results_df = pl.DataFrame(top_results)
                st.dataframe(results_df.to_pandas(), use_container_width=True)
                
                best = cf_engine.get_best_policy(results)
                st.success(f"✅ Best Policy: {best['exit_mode']} with entry_shift={best['entry_shift']}, PnL={best['pnl_pct']:.2%}")

# =======================
# TAB 3: Model Manager
# =======================
with tab3:
    st.markdown('<div class="tab-header"><h2>🤖 Model Manager</h2><p>Train and manage ML models</p></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Train New Model")
        
        if not st.session_state.cached_bars:
            st.info("Load data from Review tab first")
        else:
            model_type = st.radio(
                "Model Type",
                ["logistic", "lightgbm"],
                horizontal=True
            )
            
            snapshot_date = st.text_input(
                "Snapshot Date (YYYYMMDD)",
                datetime.now().strftime("%Y%m%d")
            )
            
            if st.button("🚀 Train Model", use_container_width=True):
                with st.spinner("Training model..."):
                    try:
                        # Get data
                        symbol = list(st.session_state.cached_bars.keys())[0]
                        bars_df = st.session_state.cached_bars[symbol]
                        
                        # Create labels
                        labeler = TripleBarrierLabeler(
                            st.session_state.config,
                            st.session_state.costs_config
                        )
                        bars_df = labeler.create_meta_labels(bars_df)
                        
                        # Train
                        engineer = FeatureEngineer(st.session_state.config)
                        feature_cols = engineer.get_feature_columns()
                        
                        trainer = MLTrainer(st.session_state.config)
                        metadata = trainer.train_and_save(
                            bars_df,
                            feature_cols,
                            model_type,
                            snapshot_date
                        )
                        
                        if 'error' not in metadata:
                            st.success(f"✅ Model trained! AUC: {metadata['metrics']['auc']:.3f}")
                            st.json(metadata)
                        else:
                            st.error(f"Training failed: {metadata['error']}")
                            
                    except Exception as e:
                        st.error(f"Error: {e}")
                        logger.error(f"Training error: {e}", exc_info=True)
    
    with col2:
        st.markdown("### 📦 Load Existing Model")
        
        models_dir = Path("models")
        if models_dir.exists():
            model_files = [f.stem for f in models_dir.glob("*_metadata.json")]
            
            if model_files:
                selected_model = st.selectbox("Select Model", model_files)
                
                if st.button("📥 Load Model", use_container_width=True):
                    trainer = MLTrainer(st.session_state.config)
                    model, metadata = trainer.load_model(selected_model)
                    
                    if model:
                        st.session_state.trained_model = (model, metadata)
                        st.success("✅ Model loaded!")
                        st.json(metadata)
            else:
                st.info("No trained models found")

# =======================
# TAB 4: Regime Controls
# =======================
with tab4:
    st.markdown('<div class="tab-header"><h2>🎯 Regime Controls</h2><p>Configure regime thresholds</p></div>', unsafe_allow_html=True)
    
    if st.session_state.cached_bars:
        symbol = list(st.session_state.cached_bars.keys())[0]
        bars_df = st.session_state.cached_bars[symbol]
        
        regime_detector = RegimeDetector(st.session_state.config)
        regime_stats = regime_detector.get_regime_stats(bars_df)
        
        # Regime distribution
        st.markdown("### 📊 Regime Distribution")
        
        regime_data = []
        for regime, stats in regime_stats.items():
            regime_data.append({'Regime': regime.capitalize(), 'Percentage': stats['pct']})
        
        if regime_data:
            regime_df = pl.DataFrame(regime_data)
            fig = px.pie(
                regime_df.to_pandas(),
                values='Percentage',
                names='Regime',
                color='Regime',
                color_discrete_map={
                    'Trend': '#10b981',
                    'Chop': '#64748b',
                    'Breakout': '#f59e0b',
                    'Unknown': '#94a3b8'
                }
            )
            fig.update_layout(height=400, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
        
        # Threshold controls
        st.markdown("### ⚙️ Threshold Settings")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Trend")
            trend_vwap_slope = st.slider("Min VWAP Slope", 0.0, 0.01, 0.0015, 0.0001, format="%.4f")
            trend_bars = st.slider("Min Directional Bars", 1, 10, 5)
        
        with col2:
            st.markdown("#### Chop")
            chop_vol = st.slider("Max Realized Vol", 0.0, 0.1, 0.025, 0.001, format="%.3f")
            chop_atr = st.slider("Max ATR %", 0.0, 0.05, 0.02, 0.001, format="%.3f")
        
        with col3:
            st.markdown("#### Breakout")
            breakout_vol_ratio = st.slider("Min Volume Ratio", 1.0, 5.0, 2.0, 0.1)
            breakout_atr_exp = st.slider("Min ATR Expansion", 1.0, 3.0, 1.5, 0.1)
        
        if st.button("💾 Save Configuration", use_container_width=True):
            # Update config
            config = st.session_state.config
            config['regime']['trend']['min_vwap_slope'] = trend_vwap_slope
            config['regime']['trend']['min_directional_bars'] = trend_bars
            config['regime']['chop']['max_realized_vol'] = chop_vol
            config['regime']['chop']['max_atr_pct'] = chop_atr
            config['regime']['breakout']['min_volume_ratio'] = breakout_vol_ratio
            config['regime']['breakout']['min_atr_expansion'] = breakout_atr_exp
            
            # Save to file
            with open('config/config.yaml', 'w') as f:
                yaml.dump(config, f)
            
            st.success("✅ Configuration saved!")
    else:
        st.info("Run a review first to see regime data")

# =======================
# TAB 5: Logs
# =======================
with tab5:
    st.markdown('<div class="tab-header"><h2>📋 System Logs</h2><p>View application logs</p></div>', unsafe_allow_html=True)
    
    log_file = Path("logs/app.log")
    
    if log_file.exists():
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        st.text_area("Logs", log_content, height=600)
        
        if st.button("🗑️ Clear Logs"):
            with open(log_file, 'w') as f:
                f.write("")
            st.success("Logs cleared!")
            st.rerun()
    else:
        st.info("No logs found")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; padding: 2rem;">
    <p><strong>Daily Review Machine</strong> • Intelligent Trading Analysis • Powered by ML</p>
</div>
""", unsafe_allow_html=True)
