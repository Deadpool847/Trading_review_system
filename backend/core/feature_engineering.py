"""Feature Engineering Module - Vectorized with Polars"""
import polars as pl
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Vectorized feature engineering using Polars"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.atr_period = config.get('features', {}).get('atr_period', 14)
        self.vol_lookback = config.get('features', {}).get('vol_lookback', [2, 5, 10])
        self.vwap_slope_bars = config.get('features', {}).get('vwap_slope_bars', 10)
    
    def compute_features(self, df: pl.DataFrame, base_score: float = None) -> pl.DataFrame:
        """Compute all features vectorized
        
        Args:
            df: Polars DataFrame with OHLCV + vwap
            base_score: Optional base score to pass through
            
        Returns:
            DataFrame with additional feature columns
        """
        if len(df) < self.atr_period + max(self.vol_lookback) + self.vwap_slope_bars:
            logger.warning("Insufficient data for feature computation")
            return df
        
        try:
            # Sort by timestamp
            df = df.sort('ts')
            
            # 1. VWAP distance and slope
            df = df.with_columns([
                # VWAP distance (normalized by price)
                ((pl.col('c') - pl.col('vwap')) / pl.col('vwap')).alias('vwap_dist'),
                
                # VWAP slope over N bars
                (pl.col('vwap').diff(self.vwap_slope_bars) / pl.col('vwap').shift(self.vwap_slope_bars)).alias('vwap_slope')
            ])
            
            # 2. ATR (Average True Range)
            df = df.with_columns([
                # True Range
                pl.max_horizontal([
                    pl.col('h') - pl.col('l'),
                    (pl.col('h') - pl.col('c').shift(1)).abs(),
                    (pl.col('l') - pl.col('c').shift(1)).abs()
                ]).alias('tr')
            ])
            
            df = df.with_columns([
                # ATR and ATR %
                pl.col('tr').rolling_mean(self.atr_period).alias('atr'),
            ])
            
            df = df.with_columns([
                (pl.col('atr') / pl.col('c')).alias('atr_pct')
            ])
            
            # 3. Realized Volatility (rolling std of returns)
            df = df.with_columns([
                pl.col('c').pct_change().alias('returns')
            ])
            
            df = df.with_columns([
                pl.col('returns').rolling_std(self.atr_period).alias('realized_vol')
            ])
            
            # 4. Volume ratios
            for lookback in self.vol_lookback:
                df = df.with_columns([
                    (pl.col('v') / pl.col('v').rolling_mean(lookback)).alias(f'volume_ratio_{lookback}')
                ])
            
            # 5. Time-of-day features (sin/cos encoding)
            df = df.with_columns([
                pl.col('ts').dt.hour().alias('hour'),
                pl.col('ts').dt.minute().alias('minute')
            ])
            
            df = df.with_columns([
                # Hour of day encoded as sin/cos
                (2 * np.pi * pl.col('hour') / 24).sin().alias('hour_sin'),
                (2 * np.pi * pl.col('hour') / 24).cos().alias('hour_cos'),
                
                # Session phase: 0=pre-market, 1=open, 2=mid, 3=close
                pl.when(pl.col('hour') < 9).then(0)
                  .when((pl.col('hour') == 9) & (pl.col('minute') < 30)).then(1)
                  .when(pl.col('hour') < 14).then(2)
                  .when(pl.col('hour') >= 15).then(3)
                  .otherwise(2)
                  .alias('session_phase')
            ])
            
            # 6. Base score passthrough
            if base_score is not None:
                df = df.with_columns([
                    pl.lit(base_score).alias('base_score')
                ])
            
            # Fill nulls from rolling operations
            df = df.fill_null(strategy='forward')
            df = df.fill_nan(0)
            
            logger.info(f"Computed features for {len(df)} bars")
            return df
            
        except Exception as e:
            logger.error(f"Error computing features: {e}")
            return df
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature column names"""
        features = [
            'vwap_dist', 'vwap_slope',
            'atr_pct', 'realized_vol',
            'hour_sin', 'hour_cos', 'session_phase'
        ]
        
        # Add volume ratios
        for lookback in self.vol_lookback:
            features.append(f'volume_ratio_{lookback}')
        
        # Add base_score if exists
        features.append('base_score')
        
        return features
