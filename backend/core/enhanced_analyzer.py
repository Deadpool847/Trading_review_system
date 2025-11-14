"""Enhanced Stock Analyzer with Advanced Metrics"""
import polars as pl
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class EnhancedAnalyzer:
    """Ruthlessly advanced stock analysis"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def compute_advanced_metrics(self, df: pl.DataFrame) -> Dict:
        """Compute comprehensive technical metrics"""
        if len(df) < 20:
            return {}
        
        try:
            metrics = {}
            
            # === PRICE METRICS ===
            first_close = df[0, 'c']
            last_close = df[-1, 'c']
            high_price = df['h'].max()
            low_price = df['l'].min()
            
            metrics['price_change_pct'] = ((last_close - first_close) / first_close) * 100
            metrics['price_range_pct'] = ((high_price - low_price) / low_price) * 100
            
            # === VOLATILITY METRICS ===
            returns = df['c'].pct_change()
            metrics['daily_volatility'] = returns.std()
            metrics['annualized_volatility'] = returns.std() * np.sqrt(252)
            
            # Parkinson's volatility (using high-low range)
            if 'h' in df.columns and 'l' in df.columns:
                hl_ratio = (df['h'] / df['l']).log()
                metrics['parkinson_volatility'] = np.sqrt((1 / (4 * len(df) * np.log(2))) * (hl_ratio ** 2).sum())
            
            # === MOMENTUM INDICATORS ===
            # RSI (14-period)
            delta = df['c'].diff()
            gain = delta.clip(lower_bound=0)
            loss = (-delta).clip(lower_bound=0)
            avg_gain = gain.rolling_mean(14)
            avg_loss = loss.rolling_mean(14)
            rs = avg_gain / avg_loss
            metrics['rsi'] = 100 - (100 / (1 + rs[-1])) if len(rs) > 0 and rs[-1] is not None else 50
            
            # MACD
            ema12 = df['c'].ewm_mean(span=12)
            ema26 = df['c'].ewm_mean(span=26)
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm_mean(span=9)
            metrics['macd'] = macd_line[-1] if len(macd_line) > 0 else 0
            metrics['macd_signal'] = signal_line[-1] if len(signal_line) > 0 else 0
            metrics['macd_histogram'] = metrics['macd'] - metrics['macd_signal']
            
            # === VOLUME METRICS ===
            metrics['avg_volume'] = df['v'].mean()
            metrics['volume_stddev'] = df['v'].std()
            recent_volume = df['v'].tail(10).mean()
            metrics['volume_trend'] = (recent_volume / metrics['avg_volume']) - 1
            
            # On-Balance Volume (OBV)
            obv = 0
            obv_list = []
            for i in range(len(df)):
                if i == 0:
                    obv_list.append(df[i, 'v'])
                else:
                    if df[i, 'c'] > df[i-1, 'c']:
                        obv += df[i, 'v']
                    elif df[i, 'c'] < df[i-1, 'c']:
                        obv -= df[i, 'v']
                    obv_list.append(obv)
            metrics['obv'] = obv_list[-1]
            metrics['obv_trend'] = "BULLISH" if obv_list[-1] > obv_list[-10] else "BEARISH"
            
            # === VWAP ANALYSIS ===
            current_price = last_close
            current_vwap = df[-1, 'vwap']
            metrics['vwap_distance_pct'] = ((current_price - current_vwap) / current_vwap) * 100
            
            # VWAP bands
            vwap_std = df['vwap'].std()
            metrics['vwap_upper_band'] = current_vwap + (2 * vwap_std)
            metrics['vwap_lower_band'] = current_vwap - (2 * vwap_std)
            
            # === SUPPORT & RESISTANCE ===
            # Recent pivot points
            recent_highs = df['h'].tail(50).to_list() if len(df) >= 50 else df['h'].to_list()
            recent_lows = df['l'].tail(50).to_list() if len(df) >= 50 else df['l'].to_list()
            
            metrics['resistance_1'] = max(recent_highs)
            metrics['resistance_2'] = sorted(recent_highs, reverse=True)[1] if len(recent_highs) > 1 else metrics['resistance_1']
            
            metrics['support_1'] = min(recent_lows)
            metrics['support_2'] = sorted(recent_lows)[1] if len(recent_lows) > 1 else metrics['support_1']
            
            # === TREND STRENGTH ===
            # ADX (Average Directional Index) - Simplified
            try:
                high_arr = df['h'].to_numpy()
                low_arr = df['l'].to_numpy()
                close_arr = df['c'].to_numpy()
                
                # True Range
                tr_arr = np.maximum(
                    high_arr - low_arr,
                    np.maximum(
                        np.abs(high_arr - np.roll(close_arr, 1)),
                        np.abs(low_arr - np.roll(close_arr, 1))
                    )
                )
                
                # Simple ADX proxy using ATR and trend consistency
                atr_14 = np.convolve(tr_arr, np.ones(14)/14, mode='valid')
                price_change = np.diff(close_arr)
                trend_consistency = np.convolve(np.abs(price_change), np.ones(14)/14, mode='valid')
                
                if len(atr_14) > 0 and len(trend_consistency) > 0:
                    adx_proxy = min(100, (trend_consistency[-1] / atr_14[-1] * 100)) if atr_14[-1] > 0 else 25
                    metrics['adx'] = adx_proxy
                else:
                    metrics['adx'] = 25
            except:
                metrics['adx'] = 25
            
            # === MARKET SENTIMENT ===
            # Bullish/Bearish bar count
            bullish_bars = len(df.filter(pl.col('c') > pl.col('o')))
            bearish_bars = len(df.filter(pl.col('c') < pl.col('o')))
            metrics['bullish_pct'] = (bullish_bars / len(df)) * 100
            metrics['bearish_pct'] = (bearish_bars / len(df)) * 100
            
            # Momentum classification
            if metrics['rsi'] > 70:
                metrics['momentum_state'] = "OVERBOUGHT"
            elif metrics['rsi'] < 30:
                metrics['momentum_state'] = "OVERSOLD"
            elif metrics['macd_histogram'] > 0 and metrics['adx'] > 25:
                metrics['momentum_state'] = "STRONG_BULLISH"
            elif metrics['macd_histogram'] < 0 and metrics['adx'] > 25:
                metrics['momentum_state'] = "STRONG_BEARISH"
            else:
                metrics['momentum_state'] = "NEUTRAL"
            
            logger.info(f"✅ Computed {len(metrics)} advanced metrics")
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing advanced metrics: {e}")
            return {}
    
    def generate_trading_signals(self, df: pl.DataFrame, metrics: Dict) -> List[Dict]:
        """Generate actionable trading signals"""
        signals = []
        
        try:
            # === ENTRY SIGNALS ===
            
            # 1. RSI Oversold + MACD Bullish Cross
            if metrics.get('rsi', 50) < 30 and metrics.get('macd_histogram', 0) > 0:
                signals.append({
                    'type': 'BUY',
                    'strength': 'STRONG',
                    'reason': 'RSI Oversold + MACD Bullish Crossover',
                    'confidence': 85
                })
            
            # 2. Price at VWAP Lower Band
            current_price = df[-1, 'c']
            vwap_lower = metrics.get('vwap_lower_band', 0)
            if vwap_lower > 0 and current_price <= vwap_lower * 1.005:
                signals.append({
                    'type': 'BUY',
                    'strength': 'MEDIUM',
                    'reason': 'Price near VWAP lower band (mean reversion)',
                    'confidence': 70
                })
            
            # 3. Strong Trend + Momentum
            if metrics.get('adx', 0) > 25 and metrics.get('momentum_state') == "STRONG_BULLISH":
                signals.append({
                    'type': 'BUY',
                    'strength': 'STRONG',
                    'reason': 'Strong uptrend with high ADX',
                    'confidence': 80
                })
            
            # === EXIT SIGNALS ===
            
            # 1. RSI Overbought
            if metrics.get('rsi', 50) > 70:
                signals.append({
                    'type': 'SELL',
                    'strength': 'MEDIUM',
                    'reason': 'RSI Overbought (> 70)',
                    'confidence': 65
                })
            
            # 2. Price at Resistance
            resistance = metrics.get('resistance_1', 0)
            if resistance > 0 and current_price >= resistance * 0.995:
                signals.append({
                    'type': 'SELL',
                    'strength': 'MEDIUM',
                    'reason': 'Price at resistance level',
                    'confidence': 70
                })
            
            # 3. MACD Bearish Cross
            if metrics.get('macd_histogram', 0) < 0 and metrics.get('macd', 0) < metrics.get('macd_signal', 0):
                signals.append({
                    'type': 'SELL',
                    'strength': 'MEDIUM',
                    'reason': 'MACD Bearish Crossover',
                    'confidence': 75
                })
            
            # === HOLD SIGNALS ===
            
            if not signals:
                signals.append({
                    'type': 'HOLD',
                    'strength': 'NEUTRAL',
                    'reason': 'No strong signals detected',
                    'confidence': 50
                })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []
    
    def risk_assessment(self, df: pl.DataFrame, metrics: Dict) -> Dict:
        """Assess risk levels"""
        risk = {}
        
        try:
            # Volatility risk
            vol = metrics.get('daily_volatility', 0)
            if vol > 0.03:
                risk['volatility'] = 'HIGH'
            elif vol > 0.015:
                risk['volatility'] = 'MEDIUM'
            else:
                risk['volatility'] = 'LOW'
            
            # Momentum risk
            momentum_state = metrics.get('momentum_state', 'NEUTRAL')
            if momentum_state in ['OVERBOUGHT', 'OVERSOLD']:
                risk['momentum'] = 'HIGH'
            elif momentum_state in ['STRONG_BULLISH', 'STRONG_BEARISH']:
                risk['momentum'] = 'MEDIUM'
            else:
                risk['momentum'] = 'LOW'
            
            # Volume risk
            volume_trend = metrics.get('volume_trend', 0)
            if abs(volume_trend) > 0.5:
                risk['volume'] = 'HIGH'
            elif abs(volume_trend) > 0.2:
                risk['volume'] = 'MEDIUM'
            else:
                risk['volume'] = 'LOW'
            
            # Overall risk score (1-10)
            risk_scores = {
                'LOW': 2,
                'MEDIUM': 5,
                'HIGH': 8
            }
            
            total_risk = sum([risk_scores.get(v, 5) for v in risk.values()])
            risk['overall_score'] = total_risk / len(risk)
            
            if risk['overall_score'] > 6:
                risk['overall'] = 'HIGH'
            elif risk['overall_score'] > 4:
                risk['overall'] = 'MEDIUM'
            else:
                risk['overall'] = 'LOW'
            
            return risk
            
        except Exception as e:
            logger.error(f"Error assessing risk: {e}")
            return {'overall': 'UNKNOWN'}
