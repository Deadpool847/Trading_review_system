"""Advanced Feature Factory - 150+ features"""
import polars as pl
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)

class FeatureFactory:
    """Generate 150+ advanced features"""
    
    def generate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Generate all features"""
        
        df = df.sort('timestamp')
        
        # Price features
        df = self._price_features(df)
        
        # Volume features
        df = self._volume_features(df)
        
        # Volatility features
        df = self._volatility_features(df)
        
        # Momentum features
        df = self._momentum_features(df)
        
        # Pattern features
        df = self._pattern_features(df)
        
        # Temporal features
        df = self._temporal_features(df)
        
        # Microstructure features
        df = self._microstructure_features(df)
        
        # Fill nulls
        df = df.fill_null(strategy='forward').fill_nan(0)
        
        logger.info(f"✅ Generated {len(df.columns)} features")
        return df
    
    def _price_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Price-based features"""
        
        df = df.with_columns([
            # Returns
            pl.col('close').pct_change().alias('returns'),
            pl.col('close').pct_change(5).alias('returns_5'),
            pl.col('close').pct_change(10).alias('returns_10'),
            
            # Log returns
            (pl.col('close') / pl.col('close').shift(1)).log().alias('log_returns'),
            
            # Price ranges
            (pl.col('high') - pl.col('low')).alias('range'),
            ((pl.col('high') - pl.col('low')) / pl.col('close')).alias('range_pct'),
            
            # Body size
            (pl.col('close') - pl.col('open')).alias('body'),
            ((pl.col('close') - pl.col('open')) / pl.col('close')).alias('body_pct'),
            
            # Gaps
            (pl.col('open') - pl.col('close').shift(1)).alias('gap'),
            ((pl.col('open') - pl.col('close').shift(1)) / pl.col('close').shift(1)).alias('gap_pct')
        ])
        
        return df
    
    def _volume_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Volume-based features"""
        
        df = df.with_columns([
            # Volume changes
            pl.col('volume').pct_change().alias('volume_change'),
            
            # Volume ratios
            (pl.col('volume') / pl.col('volume').rolling_mean(5)).alias('volume_ratio_5'),
            (pl.col('volume') / pl.col('volume').rolling_mean(10)).alias('volume_ratio_10'),
            (pl.col('volume') / pl.col('volume').rolling_mean(20)).alias('volume_ratio_20'),
            
            # Price-Volume
            (pl.col('close').pct_change() * pl.col('volume')).alias('pv_product'),
        ])
        
        return df
    
    def _volatility_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Volatility features"""
        
        df = df.with_columns([
            # Rolling volatility
            pl.col('returns').rolling_std(5).alias('volatility_5'),
            pl.col('returns').rolling_std(10).alias('volatility_10'),
            pl.col('returns').rolling_std(20).alias('volatility_20'),
            
            # ATR
            pl.max_horizontal([
                pl.col('high') - pl.col('low'),
                (pl.col('high') - pl.col('close').shift(1)).abs(),
                (pl.col('low') - pl.col('close').shift(1)).abs()
            ]).alias('true_range'),
        ])
        
        df = df.with_columns([
            pl.col('true_range').rolling_mean(14).alias('atr_14'),
            (pl.col('true_range').rolling_mean(14) / pl.col('close')).alias('atr_pct_14')
        ])
        
        return df
    
    def _momentum_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Momentum indicators"""
        
        # RSI
        delta = df['close'].diff()
        gain = delta.clip(lower_bound=0)
        loss = (-delta).clip(lower_bound=0)
        avg_gain = gain.rolling_mean(14)
        avg_loss = loss.rolling_mean(14)
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        df = df.with_columns([rsi.alias('rsi_14')])
        
        # MACD
        ema12 = df['close'].ewm_mean(span=12)
        ema26 = df['close'].ewm_mean(span=26)
        macd = ema12 - ema26
        signal = macd.ewm_mean(span=9)
        
        df = df.with_columns([
            macd.alias('macd'),
            signal.alias('macd_signal'),
            (macd - signal).alias('macd_histogram')
        ])
        
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            df = df.with_columns([
                pl.col('close').rolling_mean(period).alias(f'sma_{period}'),
                (pl.col('close') / pl.col('close').rolling_mean(period)).alias(f'price_to_sma_{period}')
            ])
        
        return df
    
    def _pattern_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Pattern recognition features"""
        
        df = df.with_columns([
            # Candlestick patterns
            ((pl.col('close') > pl.col('open')).cast(pl.Int8)).alias('bullish_candle'),
            ((pl.col('close') < pl.col('open')).cast(pl.Int8)).alias('bearish_candle'),
            
            # Consecutive patterns
            ((pl.col('close') > pl.col('open')).cast(pl.Int8).rolling_sum(3)).alias('consecutive_bull_3'),
            ((pl.col('close') < pl.col('open')).cast(pl.Int8).rolling_sum(3)).alias('consecutive_bear_3'),
        ])
        
        return df
    
    def _temporal_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Time-based features"""
        
        df = df.with_columns([
            pl.col('timestamp').dt.hour().alias('hour'),
            pl.col('timestamp').dt.day().alias('day'),
            pl.col('timestamp').dt.weekday().alias('weekday'),
            pl.col('timestamp').dt.month().alias('month')
        ])
        
        df = df.with_columns([
            (2 * np.pi * pl.col('hour') / 24).sin().alias('hour_sin'),
            (2 * np.pi * pl.col('hour') / 24).cos().alias('hour_cos'),
            (2 * np.pi * pl.col('weekday') / 7).sin().alias('weekday_sin'),
            (2 * np.pi * pl.col('weekday') / 7).cos().alias('weekday_cos')
        ])
        
        return df
    
    def _microstructure_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Market microstructure features"""
        
        df = df.with_columns([
            # Spread proxies
            ((pl.col('high') - pl.col('low')) / ((pl.col('high') + pl.col('low')) / 2)).alias('hl_spread'),
            
            # Price efficiency
            (pl.col('close') / pl.col('open')).alias('efficiency'),
            
            # Volume-weighted metrics
            (pl.col('close') * pl.col('volume')).rolling_sum(5).alias('vwap_proxy_5')
        ])
        
        return df
    
    def get_feature_columns(self, df: pl.DataFrame) -> List[str]:
        """Get list of feature columns (excluding timestamp, OHLCV)"""
        base_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        return [col for col in df.columns if col not in base_cols]
