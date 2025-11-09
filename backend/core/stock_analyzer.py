"""Advanced Stock Analysis Module"""
import polars as pl
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class StockAnalyzer:
    """Comprehensive stock analysis with actionable insights"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def analyze_price_action(self, df: pl.DataFrame) -> Dict:
        """Analyze overall price action and trends"""
        if len(df) == 0:
            return {}
        
        try:
            recent_bars = min(20, len(df))
            recent_df = df.tail(recent_bars)
            
            # Price movement
            first_close = df[0, 'c']
            last_close = df[-1, 'c']
            price_change_pct = ((last_close - first_close) / first_close) * 100
            
            # Volatility
            returns = df['c'].pct_change()
            volatility = returns.std()
            
            # Recent momentum
            recent_close = recent_df['c'].to_list()
            momentum = "BULLISH" if recent_close[-1] > recent_close[0] else "BEARISH"
            
            # VWAP analysis
            current_price = last_close
            current_vwap = df[-1, 'vwap']
            vwap_position = "ABOVE" if current_price > current_vwap else "BELOW"
            vwap_distance_pct = ((current_price - current_vwap) / current_vwap) * 100
            
            # Volume analysis
            avg_volume = df['v'].mean()
            recent_volume = recent_df['v'].mean()
            volume_trend = "INCREASING" if recent_volume > avg_volume * 1.2 else "DECREASING" if recent_volume < avg_volume * 0.8 else "STABLE"
            
            # Support and Resistance (simple)
            highs = df['h'].to_list()
            lows = df['l'].to_list()
            resistance = max(highs[-20:]) if len(highs) >= 20 else max(highs)
            support = min(lows[-20:]) if len(lows) >= 20 else min(lows)
            
            analysis = {
                'price_change_pct': price_change_pct,
                'momentum': momentum,
                'volatility': volatility,
                'vwap_position': vwap_position,
                'vwap_distance_pct': vwap_distance_pct,
                'volume_trend': volume_trend,
                'current_price': current_price,
                'resistance': resistance,
                'support': support,
                'avg_volume': avg_volume
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in price action analysis: {e}")
            return {}
    
    def generate_insights(self, df: pl.DataFrame, analysis: Dict, regime: str = None) -> List[str]:
        """Generate actionable trading insights"""
        insights = []
        
        if not analysis:
            return ["Insufficient data for analysis"]
        
        try:
            # Price trend insight
            if abs(analysis['price_change_pct']) > 2:
                direction = "up" if analysis['price_change_pct'] > 0 else "down"
                insights.append(f"🎯 Strong {direction}ward movement: {abs(analysis['price_change_pct']):.2f}% change")
            
            # VWAP insight
            if abs(analysis['vwap_distance_pct']) > 1:
                if analysis['vwap_position'] == "ABOVE":
                    insights.append(f"📈 Price trading {abs(analysis['vwap_distance_pct']):.2f}% above VWAP - Bullish signal")
                else:
                    insights.append(f"📉 Price trading {abs(analysis['vwap_distance_pct']):.2f}% below VWAP - Bearish pressure")
            else:
                insights.append("⚖️ Price near VWAP - Balanced market")
            
            # Volatility insight
            if analysis['volatility'] > 0.02:
                insights.append(f"⚠️ High volatility detected ({analysis['volatility']:.3f}) - Use wider stops")
            elif analysis['volatility'] < 0.01:
                insights.append(f"🔒 Low volatility ({analysis['volatility']:.3f}) - Tight range, breakout potential")
            
            # Volume insight
            if analysis['volume_trend'] == "INCREASING":
                insights.append("📊 Volume increasing - Confirms price movement")
            elif analysis['volume_trend'] == "DECREASING":
                insights.append("📉 Volume declining - Weak momentum, caution advised")
            
            # Support/Resistance levels
            current = analysis['current_price']
            resistance = analysis['resistance']
            support = analysis['support']
            
            distance_to_resistance = ((resistance - current) / current) * 100
            distance_to_support = ((current - support) / current) * 100
            
            if distance_to_resistance < 1:
                insights.append(f"🚧 Near resistance at ₹{resistance:.2f} - Watch for rejection or breakout")
            
            if distance_to_support < 1:
                insights.append(f"🛡️ Near support at ₹{support:.2f} - Potential bounce zone")
            
            # Regime-specific insights
            if regime == 'trend':
                insights.append("🚀 TREND regime detected - Momentum strategies favored")
                if analysis['momentum'] == "BULLISH":
                    insights.append("✅ Bullish trend - Consider long positions on pullbacks")
                else:
                    insights.append("⚠️ Bearish trend - Avoid longs, consider shorts")
            
            elif regime == 'chop':
                insights.append("〰️ CHOP regime detected - Range-bound, mean-reversion strategies")
                insights.append(f"🎯 Trade between support (₹{support:.2f}) and resistance (₹{resistance:.2f})")
            
            elif regime == 'breakout':
                insights.append("💥 BREAKOUT regime detected - High volume + volatility")
                insights.append("⚡ Quick exits recommended - High risk/reward setup")
            
            # Overall recommendation
            if len(insights) == 0:
                insights.append("📊 Market in consolidation - Wait for clear setup")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return ["Error generating insights"]
    
    def merge_with_trade_logs(
        self,
        df: pl.DataFrame,
        scanner_results: List[Dict],
        trade_entries: List[Dict],
        symbol: str
    ) -> Dict:
        """Merge chart data with trade logs for enhanced visualization
        
        Returns:
            Dict with markers for entries, exits, scanner scores
        """
        markers = {
            'entry_points': [],
            'target_lines': [],
            'stop_lines': [],
            'scanner_scores': []
        }
        
        if len(df) == 0:
            return markers
        
        try:
            # Filter relevant trades for this symbol
            symbol_trades = [t for t in trade_entries if t['symbol'] == symbol]
            
            for trade in symbol_trades:
                if trade['timestamp']:
                    # Find closest bar to trade timestamp
                    df_with_ts = df.with_columns([
                        (pl.col('ts') - pl.lit(trade['timestamp'])).abs().alias('ts_diff')
                    ])
                    closest_idx = df_with_ts['ts_diff'].arg_min()
                    
                    if closest_idx is not None:
                        closest_bar = df[closest_idx]
                        
                        markers['entry_points'].append({
                            'ts': closest_bar['ts'],
                            'price': trade['entry'],
                            'qty': trade['qty'],
                            'type': 'LONG'
                        })
                        
                        if trade['target']:
                            markers['target_lines'].append({
                                'price': trade['target'],
                                'label': f"Target: ₹{trade['target']:.2f}"
                            })
                        
                        if trade['stop_loss']:
                            markers['stop_lines'].append({
                                'price': trade['stop_loss'],
                                'label': f"SL: ₹{trade['stop_loss']:.2f}"
                            })
            
            # Add scanner scores
            symbol_scans = [s for s in scanner_results if s['symbol'] == symbol]
            for scan in symbol_scans:
                markers['scanner_scores'].append({
                    'score': scan['score'],
                    'ltp': scan['ltp'],
                    'atr_pct': scan['atr_pct'],
                    'vwap_dev': scan['vwap_dev'],
                    'volume': scan['volume']
                })
            
            logger.info(f"✅ Merged {len(markers['entry_points'])} entries, {len(markers['scanner_scores'])} scanner results")
            return markers
            
        except Exception as e:
            logger.error(f"Error merging trade logs: {e}")
            return markers
    
    def generate_summary(
        self,
        symbol: str,
        df: pl.DataFrame,
        analysis: Dict,
        insights: List[str],
        regime: str = None
    ) -> str:
        """Generate comprehensive text summary"""
        if len(df) == 0:
            return f"No data available for {symbol}"
        
        try:
            summary = f"""
# 📊 Stock Analysis: {symbol}

## 📈 Price Action
- **Current Price:** ₹{analysis.get('current_price', 0):.2f}
- **Period Change:** {analysis.get('price_change_pct', 0):.2f}%
- **Momentum:** {analysis.get('momentum', 'N/A')}
- **Volatility:** {analysis.get('volatility', 0):.3f}

## 🎯 Key Levels
- **Resistance:** ₹{analysis.get('resistance', 0):.2f}
- **Support:** ₹{analysis.get('support', 0):.2f}
- **VWAP:** ₹{df[-1, 'vwap']:.2f} ({analysis.get('vwap_position', 'N/A')})

## 📊 Volume Analysis
- **Avg Volume:** {int(analysis.get('avg_volume', 0)):,}
- **Trend:** {analysis.get('volume_trend', 'N/A')}

## 🎯 Market Regime
{regime.upper() if regime else 'UNKNOWN'}

## 💡 Actionable Insights

{chr(10).join([f'{i+1}. {insight}' for i, insight in enumerate(insights)])}

---
*Analysis based on {len(df)} candles*
"""
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Error generating summary for {symbol}"
