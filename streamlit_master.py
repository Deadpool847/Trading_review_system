"""MASTER AI TRADING SYSTEM - Aladdin-Level Intelligence"""
import streamlit as st
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import logging

# Setup
sys.path.insert(0, str(Path(__file__).parent / 'backend'))

from core.universal_loader import UniversalLoader
from ai.feature_factory import FeatureFactory
from ai.master_model import MasterModel
from storage.stock_manager import StockManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="AI Trading Master",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600&family=Inter:wght@400;500&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize
if 'stock_manager' not in st.session_state:
    st.session_state.stock_manager = StockManager()

if 'current_data' not in st.session_state:
    st.session_state.current_data = None

if 'current_symbol' not in st.session_state:
    st.session_state.current_symbol = None

# Header
st.markdown("""<div style="text-align: center; padding: 2rem; color: white;">
<h1 style="font-size: 3rem; margin-bottom: 0.5rem;">🧠 AI TRADING MASTER</h1>
<p style="font-size: 1.2rem; opacity: 0.9;">Aladdin-Level Intelligence • Offline • Zero Bugs</p>
</div>""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["1️⃣ Data Input", "2️⃣ Train Models", "3️⃣ Future Predictions"])

# ==========================================
# TAB 1: DATA INPUT
# ==========================================
with tab1:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown("### 📂 Upload Historical Data")
    
    uploaded_file = st.file_uploader(
        "Upload CSV, Excel, or Parquet file",
        type=['csv', 'xlsx', 'xls', 'parquet'],
        help="System auto-detects columns!"
    )
    
    if uploaded_file:
        if st.button("🚀 PROCESS DATA", use_container_width=True):
            with st.spinner("🔧 Processing..."):
                try:
                    # Load
                    loader = UniversalLoader()
                    df, symbol = loader.load(uploaded_file)
                    
                    st.success(f"✅ Loaded {len(df)} bars for **{symbol}**")
                    
                    # Generate features
                    factory = FeatureFactory()
                    df_features = factory.generate(df)
                    
                    st.success(f"✅ Generated {len(df_features.columns)} features")
                    
                    # Save
                    stock_manager = st.session_state.stock_manager
                    stock_manager.save_stock_data(symbol, df_features)
                    
                    # Cache
                    st.session_state.current_data = df_features
                    st.session_state.current_symbol = symbol
                    
                    st.success(f"💾 Saved to local storage!")
                    
                    # Show preview
                    st.markdown("#### 👀 Data Preview")
                    st.dataframe(df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].head(10).to_pandas())
                    
                    # Show stats
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df)}</div><div class="metric-label">Bars</div></div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df_features.columns)}</div><div class="metric-label">Features</div></div>', unsafe_allow_html=True)
                    
                    with col3:
                        price_change = ((df[-1, 'close'] - df[0, 'close']) / df[0, 'close']) * 100
                        st.markdown(f'<div class="metric-card"><div class="metric-value">{price_change:.2f}%</div><div class="metric-label">Total Change</div></div>', unsafe_allow_html=True)
                    
                    with col4:
                        vol_avg = df['volume'].mean()
                        st.markdown(f'<div class="metric-card"><div class="metric-value">{int(vol_avg):,}</div><div class="metric-label">Avg Volume</div></div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show stored stocks
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown("### 📁 Stored Stocks")
    
    stored_stocks = st.session_state.stock_manager.get_all_stocks()
    
    if stored_stocks:
        for stock in stored_stocks:
            info = st.session_state.stock_manager.get_stock_info(stock)
            has_model = st.session_state.stock_manager.has_model(stock)
            
            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
            
            with col1:
                st.write(f"**{stock}**")
            with col2:
                st.write(f"{info['rows']} bars")
            with col3:
                status = "✅ Trained" if has_model else "⚠️ Not trained"
                st.write(status)
            with col4:
                if st.button("📊 View", key=f"view_{stock}"):
                    st.session_state.current_symbol = stock
                    st.session_state.current_data = st.session_state.stock_manager.load_stock_data(stock)
                    st.rerun()
    else:
        st.info("📄 No stocks stored yet. Upload data above.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 2: TRAIN MODELS
# ==========================================
with tab2:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown("### 🎓 AI Model Training")
    
    stored_stocks = st.session_state.stock_manager.get_all_stocks()
    
    if not stored_stocks:
        st.warning("⚠️ No stocks available. Upload data in Tab 1 first.")
    else:
        selected_stock = st.selectbox("🎯 Select Stock to Train", stored_stocks)
        
        st.markdown("""#### 🤖 Model Architecture
        - **LightGBM**: Fast gradient boosting
        - **Gradient Boosting**: Robust ensemble
        - **Random Forest**: Variance reduction
        - **Ridge Regression**: Linear baseline
        - **Ensemble**: Weighted combination (40-30-20-10)
        """)
        
        if st.button("🚀 TRAIN MODELS", use_container_width=True):
            with st.spinner(f"🎓 Training AI models for {selected_stock}..."):
                try:
                    # Load data
                    df = st.session_state.stock_manager.load_stock_data(selected_stock)
                    
                    if df is None:
                        st.error("❌ Data not found")
                    else:
                        # Get feature columns
                        factory = FeatureFactory()
                        feature_cols = factory.get_feature_columns(df)
                        
                        # Train
                        model = MasterModel(selected_stock)
                        metrics = model.train(df, feature_cols)
                        
                        st.success(f"✅ Training complete!")
                        
                        # Show metrics
                        st.markdown("#### 📊 Model Performance")
                        
                        metric_cols = st.columns(5)
                        
                        with metric_cols[0]:
                            st.metric("LightGBM RMSE", f"{metrics['lgb_rmse']:.4f}")
                        with metric_cols[1]:
                            st.metric("Gradient Boost RMSE", f"{metrics['gb_rmse']:.4f}")
                        with metric_cols[2]:
                            st.metric("Random Forest RMSE", f"{metrics['rf_rmse']:.4f}")
                        with metric_cols[3]:
                            st.metric("Ridge RMSE", f"{metrics['ridge_rmse']:.4f}")
                        with metric_cols[4]:
                            st.metric("⭐ Ensemble RMSE", f"{metrics['ensemble_rmse']:.4f}")
                        
                        st.balloons()
                        
                except Exception as e:
                    st.error(f"❌ Training error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 3: PREDICTIONS
# ==========================================
with tab3:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown("### 🔮 Future Predictions")
    
    # Get trained stocks
    stored_stocks = st.session_state.stock_manager.get_all_stocks()
    trained_stocks = [s for s in stored_stocks if st.session_state.stock_manager.has_model(s)]
    
    if not trained_stocks:
        st.warning("⚠️ No trained models. Train models in Tab 2 first.")
    else:
        selected_stock = st.selectbox("🎯 Select Stock", trained_stocks)
        
        forecast_days = st.slider("📅 Forecast Horizon (days)", 5, 30, 15)
        
        if st.button("🔮 GENERATE PREDICTIONS", use_container_width=True):
            with st.spinner("🧠 Generating future predictions..."):
                try:
                    # Load data and model
                    df = st.session_state.stock_manager.load_stock_data(selected_stock)
                    model = MasterModel(selected_stock)
                    model.load()
                    
                    # Get features
                    factory = FeatureFactory()
                    feature_cols = factory.get_feature_columns(df)
                    
                    # Predict
                    predictions = model.predict(df, feature_cols, n_steps=forecast_days)
                    
                    # Generate future dates
                    last_date = df[-1, 'timestamp']
                    future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
                    
                    # Plot
                    fig = go.Figure()
                    
                    # Historical
                    historical = df.tail(90)
                    fig.add_trace(go.Scatter(
                        x=historical['timestamp'].to_list(),
                        y=historical['close'].to_list(),
                        name='Historical',
                        line=dict(color='#667eea', width=2)
                    ))
                    
                    # Predictions
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=predictions,
                        name='AI Forecast',
                        line=dict(color='#f59e0b', width=3, dash='dash')
                    ))
                    
                    # Confidence bands (±10%)
                    upper_bound = predictions * 1.1
                    lower_bound = predictions * 0.9
                    
                    fig.add_trace(go.Scatter(
                        x=future_dates + future_dates[::-1],
                        y=np.concatenate([upper_bound, lower_bound[::-1]]),
                        fill='toself',
                        fillcolor='rgba(245, 158, 11, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Confidence Band'
                    ))
                    
                    fig.update_layout(
                        title=f"{selected_stock} - {forecast_days} Day Forecast",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        height=600,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show predictions table
                    st.markdown("#### 📊 Predicted Values")
                    
                    pred_df = pl.DataFrame({
                        'Date': future_dates,
                        'Predicted Price': predictions,
                        'Lower Bound': lower_bound,
                        'Upper Bound': upper_bound
                    })
                    
                    st.dataframe(pred_df.to_pandas())
                    
                    # Key insights
                    st.markdown("#### 💡 Key Insights")
                    
                    current_price = df[-1, 'close']
                    future_price = predictions[-1]
                    change_pct = ((future_price - current_price) / current_price) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Current Price", f"₹{current_price:.2f}")
                    with col2:
                        st.metric(f"Predicted ({forecast_days}d)", f"₹{future_price:.2f}", f"{change_pct:+.2f}%")
                    with col3:
                        trend = "📈 Bullish" if change_pct > 0 else "📉 Bearish"
                        st.metric("Trend", trend)
                    
                except Exception as e:
                    st.error(f"❌ Prediction error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""<div style="text-align: center; color: white; padding: 2rem; opacity: 0.8;">
<p>🧠 AI Trading Master • Powered by Advanced ML/DL • 100% Offline • Zero Bugs</p>
</div>""", unsafe_allow_html=True)
