"""Tab 1 - Rewired Super-Intelligent Analysis

This is the complete rewired Tab 1 implementation.
Copy this into streamlit_app.py to replace the existing Tab 1 section.
"""

# Paste this code to replace Tab 1 in streamlit_app.py:

# =======================
# TAB 1: Super-Intelligent Stock Analysis
# =======================
with tab1:
    st.markdown('<div class="tab-header"><h2>🧠 Super-Intelligent Stock Analysis</h2><p>Ruthlessly advanced - Handles ANY data format</p></div>', unsafe_allow_html=True)
    
    # DATA SOURCE SELECTION
    st.markdown("### 📥 Step 1: Choose Data Source")
    
    data_source = st.radio(
        "How do you want to provide data?",
        ["📁 Upload File (CSV/Excel/Parquet)", "🌐 Fetch from GrowwAPI"],
        horizontal=True,
        key="data_source"
    )
    
    st.markdown("---")
    
    # Initialize variables
    symbol_input = "UNKNOWN"
    interval_minutes = 5
    days_back = 7
    uploaded_file = None
    log_file = None
    
    # OPTION 1: FILE UPLOAD
    if data_source == "📁 Upload File (CSV/Excel/Parquet)":
        st.markdown("### 📊 Upload Historical Data")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload Historical OHLCV Data",
                type=['csv', 'xlsx', 'xls', 'parquet'],
                help="System will auto-detect columns!",
                key="uploaded_historical"
            )
            
            if uploaded_file:
                st.success(f"✅ File: {uploaded_file.name}")
        
        with col2:
            symbol_input = st.text_input("Symbol", "UNKNOWN", key="symbol_file")
            log_file = st.file_uploader("Trade Logs (Optional)", type=['log', 'txt'], key="log_file_upload")
        
        st.info("💡 Auto-detects: timestamp/ts, open/o, high/h, low/l, close/c/ltp, volume/v")
    
    # OPTION 2: GROWWAPI
    else:
        st.markdown("### 🌐 GrowwAPI Configuration")
        
        st.warning("⚠️ Implement GrowwAPI in `/app/backend/core/groww_helper.py` first!")
        
        with st.expander("🔧 Integration Guide"):
            st.code("""
# Edit: /app/backend/core/groww_helper.py
# Method: fetch_historical_data()

import growwapi as groww
client = groww.Groww()

response = client.get_historical_candles(
    exchange=groww.EXCHANGE_NSE,
    segment=groww.SEGMENT_CASH,
    groww_symbol=trading_symbol,  # Already provided
    start_time=start_time,         # Already provided
    end_time=end_time,             # Already provided
    candle_interval=your_constant
)

# Parse and return DataFrame
# See INTEGRATION_GUIDE.md for full template
            """, language="python")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            symbol_input = st.text_input("Symbol", "NSE-RELIANCE", key="symbol_api")
        with col2:
            intervals = {"1 min": 1, "5 min": 5, "15 min": 15, "30 min": 30, "1 hour": 60}
            interval_choice = st.selectbox("Interval", list(intervals.keys()), index=1, key="interval_api")
            interval_minutes = intervals[interval_choice]
        with col3:
            max_days = {1:30, 5:30, 15:90, 30:90, 60:180}.get(interval_minutes, 30)
            days_back = st.number_input(f"Days (max {max_days})", 1, max_days, min(7, max_days), key="days_api")
        with col4:
            log_file = st.file_uploader("Logs (Opt)", type=['log', 'txt'], key="log_file_api")
    
    st.markdown("---")
    
    # ANALYSIS BUTTON
    if st.button("🚀 RUN SUPER-INTELLIGENT ANALYSIS", key="run_super", use_container_width=True):
        with st.spinner("🧠 Analyzing..."):
            try:
                bars_df = None
                
                # ===== LOAD DATA =====
                if data_source == "📁 Upload File (CSV/Excel/Parquet)":
                    if not uploaded_file:
                        st.error("⚠️ Please upload a data file first!")
                        st.stop()
                    
                    st.info("📂 Loading file...")
                    validator = DataValidator()
                    
                    # Load file
                    raw_df = validator.load_file(uploaded_file)
                    st.success(f"✅ Loaded {len(raw_df)} rows, {len(raw_df.columns)} columns")
                    
                    # Normalize
                    bars_df = validator.validate_and_normalize(raw_df, symbol_input)
                    st.success(f"✅ Normalized to standard format")
                    
                    # Validate integrity
                    is_valid, issues = validator.validate_ohlc_integrity(bars_df)
                    if issues:
                        with st.expander("⚠️ Data Integrity Warnings"):
                            for issue in issues:
                                st.warning(issue)
                
                else:  # GrowwAPI
                    st.info("🌐 Fetching from GrowwAPI...")
                    
                    end_time = datetime.now()
                    start_time = end_time - timedelta(days=days_back)
                    
                    if interval_minutes < 1440:
                        start_time = start_time.replace(hour=9, minute=15, second=0)
                        end_time = end_time.replace(hour=15, minute=30, second=0)
                    
                    start_str = format_date_for_api(start_time)
                    end_str = format_date_for_api(end_time)
                    
                    groww_helper = GrowwHelper()
                    
                    try:
                        bars_df = groww_helper.fetch_historical_data(
                            trading_symbol=symbol_input,
                            start_time=start_str,
                            end_time=end_str,
                            interval_in_minutes=interval_minutes
                        )
                        st.success(f"✅ Fetched {len(bars_df)} bars from GrowwAPI")
                    except NotImplementedError as e:
                        st.error(str(e))
                        st.info("💡 **Alternative**: Use File Upload option with CSV/Excel data")
                        st.stop()
                
                # ===== PROCESS TRADE LOGS =====
                scanner_results = []
                trade_entries = []
                
                if log_file:
                    temp_path = Path("data/trades/temp_log.log")
                    temp_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(temp_path, 'wb') as f:
                        f.write(log_file.getvalue())
                    
                    ingestion = DataIngestion()
                    scanner_results, trade_entries = ingestion.load_trade_logs(str(temp_path))
                    
                    if scanner_results or trade_entries:
                        st.success(f"✅ Parsed {len(scanner_results)} scans, {len(trade_entries)} trades")
                
                # ===== FEATURE ENGINEERING =====
                if len(bars_df) < 20:
                    st.error("❌ Insufficient data (need at least 20 bars)")
                    st.stop()
                
                st.info("🔧 Computing features...")
                engineer = FeatureEngineer(st.session_state.config)
                bars_df = engineer.compute_features(bars_df)
                
                # ===== REGIME DETECTION =====
                st.info("🎯 Detecting regime...")
                regime_detector = RegimeDetector(st.session_state.config)
                bars_df = regime_detector.detect_regime(bars_df)
                
                regime_stats = regime_detector.get_regime_stats(bars_df)
                dominant_regime = max(regime_stats.items(), key=lambda x: x[1]['count'])[0] if regime_stats else 'unknown'
                
                # ===== ADVANCED ANALYSIS =====
                st.info("🧠 Running advanced analysis...")
                
                # Basic analysis
                analyzer = StockAnalyzer(st.session_state.config)
                basic_metrics = analyzer.analyze_price_action(bars_df)
                
                # Enhanced analysis
                enhanced_analyzer = EnhancedAnalyzer(st.session_state.config)
                advanced_metrics = enhanced_analyzer.compute_advanced_metrics(bars_df)
                
                # Merge metrics
                all_metrics = {**basic_metrics, **advanced_metrics}
                
                # Generate signals
                signals = enhanced_analyzer.generate_trading_signals(bars_df, all_metrics)
                
                # Risk assessment
                risk = enhanced_analyzer.risk_assessment(bars_df, all_metrics)
                
                # Basic insights
                insights = analyzer.generate_insights(bars_df, basic_metrics, dominant_regime)
                
                # Merge trade logs
                markers = analyzer.merge_with_trade_logs(bars_df, scanner_results, trade_entries, symbol_input)
                
                # Cache
                st.session_state.cached_bars[symbol_input] = bars_df
                st.session_state.analysis_cache[symbol_input] = {
                    'metrics': all_metrics,
                    'signals': signals,
                    'risk': risk,
                    'insights': insights,
                    'regime': dominant_regime,
                    'markers': markers
                }
                
                # ===== DISPLAY RESULTS =====
                st.markdown("---")
                st.markdown("## 📊 Analysis Results")
                
                # KPI Cards
                kpi_cols = st.columns(6)
                
                with kpi_cols[0]:
                    price = all_metrics.get('current_price', bars_df[-1, 'c'])
                    st.markdown(f'<div class="metric-card"><div class="metric-value">₹{price:.2f}</div><div class="metric-label">Price</div></div>', unsafe_allow_html=True)
                
                with kpi_cols[1]:
                    change = all_metrics.get('price_change_pct', 0)
                    color = "positive" if change > 0 else "negative"
                    st.markdown(f'<div class="metric-card"><div class="metric-value {color}">{change:.2f}%</div><div class="metric-label">Change</div></div>', unsafe_allow_html=True)
                
                with kpi_cols[2]:
                    momentum = all_metrics.get('momentum_state', 'NEUTRAL')
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{momentum}</div><div class="metric-label">Momentum</div></div>', unsafe_allow_html=True)
                
                with kpi_cols[3]:
                    rsi = all_metrics.get('rsi', 50)
                    rsi_color = "negative" if rsi > 70 else "positive" if rsi < 30 else ""
                    st.markdown(f'<div class="metric-card"><div class="metric-value {rsi_color}">{rsi:.1f}</div><div class="metric-label">RSI</div></div>', unsafe_allow_html=True)
                
                with kpi_cols[4]:
                    vol = all_metrics.get('daily_volatility', 0)
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{vol:.3f}</div><div class="metric-label">Volatility</div></div>', unsafe_allow_html=True)
                
                with kpi_cols[5]:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{dominant_regime.upper()}</div><div class="metric-label">Regime</div></div>', unsafe_allow_html=True)
                
                # Trading Signals
                st.markdown("---")
                st.markdown("### 🎯 Trading Signals")
                
                signal_cols = st.columns(len(signals))
                for i, signal in enumerate(signals):
                    with signal_cols[i]:
                        signal_color = "🟢" if signal['type'] == 'BUY' else "🔴" if signal['type'] == 'SELL' else "🟡"
                        st.markdown(f"""
                        **{signal_color} {signal['type']}**  
                        Strength: {signal['strength']}  
                        {signal['reason']}  
                        Confidence: {signal['confidence']}%
                        """)
                
                # Risk Assessment
                st.markdown("---")
                st.markdown("### ⚠️ Risk Assessment")
                
                risk_cols = st.columns(4)
                with risk_cols[0]:
                    st.metric("Overall Risk", risk.get('overall', 'UNKNOWN'))
                with risk_cols[1]:
                    st.metric("Volatility Risk", risk.get('volatility', 'UNKNOWN'))
                with risk_cols[2]:
                    st.metric("Momentum Risk", risk.get('momentum', 'UNKNOWN'))
                with risk_cols[3]:
                    st.metric("Volume Risk", risk.get('volume', 'UNKNOWN'))
                
                # Insights
                st.markdown("---")
                st.markdown("### 💡 Actionable Insights")
                
                insight_cols = st.columns(2)
                mid = len(insights) // 2
                
                with insight_cols[0]:
                    st.markdown("#### Trading Signals")
                    for insight in insights[:mid]:
                        st.markdown(f"- {insight}")
                
                with insight_cols[1]:
                    st.markdown("#### Market Context")
                    for insight in insights[mid:]:
                        st.markdown(f"- {insight}")
                
                # Interactive Chart
                st.markdown("---")
                st.markdown("### 📈 Interactive Chart")
                
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
                
                bars_pd = bars_df.to_pandas()
                
                # Candlestick
                fig.add_trace(go.Candlestick(x=bars_pd['ts'], open=bars_pd['o'], high=bars_pd['h'], low=bars_pd['l'], close=bars_pd['c'], name="Price"), row=1, col=1)
                
                # VWAP
                fig.add_trace(go.Scatter(x=bars_pd['ts'], y=bars_pd['vwap'], name="VWAP", line=dict(color='#f59e0b', width=2)), row=1, col=1)
                
                # Support/Resistance
                fig.add_hline(y=all_metrics.get('support_1', 0), line_dash="dot", line_color="green", annotation_text="Support", row=1, col=1)
                fig.add_hline(y=all_metrics.get('resistance_1', 0), line_dash="dot", line_color="red", annotation_text="Resistance", row=1, col=1)
                
                # Trade markers
                if markers['entry_points']:
                    entry_times = [e['ts'] for e in markers['entry_points']]
                    entry_prices = [e['price'] for e in markers['entry_points']]
                    fig.add_trace(go.Scatter(x=entry_times, y=entry_prices, mode='markers', marker=dict(size=15, color='#10b981', symbol='triangle-up'), name='Entries'), row=1, col=1)
                
                # Volume
                regime_colors = {'trend': '#10b981', 'chop': '#64748b', 'breakout': '#f59e0b', 'unknown': '#94a3b8'}
                for regime, color in regime_colors.items():
                    regime_bars = bars_pd[bars_pd['regime'] == regime]
                    if len(regime_bars) > 0:
                        fig.add_trace(go.Bar(x=regime_bars['ts'], y=regime_bars['v'], name=regime.capitalize(), marker_color=color), row=2, col=1)
                
                fig.update_layout(height=800, template='plotly_white', xaxis_rangeslider_visible=False, hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"✅ Analysis complete! {len(bars_df)} bars • {len(signals)} signals • {len(insights)} insights")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
                logger.error(f"Analysis error: {e}", exc_info=True)
                import traceback
                with st.expander("🐛 Error Details"):
                    st.code(traceback.format_exc())
