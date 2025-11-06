"""Data Ingestion Module - GrowwAPI Integration"""
import polars as pl
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any
import logging
from pathlib import Path
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

class DataIngestion:
    """Handles data fetching from GrowwAPI and caching"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_ohlcv_bars(
        self,
        groww_client,
        symbol: str,
        exchange: str,
        segment: str,
        start_time: str,
        end_time: str,
        interval_minutes: int = 5,
        use_cache: bool = True
    ) -> pl.DataFrame:
        """Fetch OHLCV bars from GrowwAPI with caching
        
        Args:
            groww_client: Authenticated GrowwAPI client
            symbol: Trading symbol (e.g., 'RELIANCE')
            exchange: Exchange code (e.g., 'NSE')
            segment: Segment code (e.g., 'CASH')
            start_time: Start timestamp 'YYYY-MM-DD HH:MM:SS'
            end_time: End timestamp 'YYYY-MM-DD HH:MM:SS'
            interval_minutes: Candle interval (5, 15, 30, etc.)
            use_cache: Whether to use cached data
            
        Returns:
            Polars DataFrame with columns: [ts, symbol, o, h, l, c, v, vwap]
        """
        cache_key = f"{symbol}_{exchange}_{interval_minutes}m_{start_time.replace(' ', '_')}_{end_time.replace(' ', '_')}.parquet"
        cache_path = self.cache_dir / cache_key
        
        # Check cache
        if use_cache and cache_path.exists():
            logger.info(f"Loading cached data for {symbol}")
            return pl.read_parquet(cache_path)
        
        try:
            logger.info(f"Fetching {symbol} from GrowwAPI: {start_time} to {end_time}")
            response = groww_client.get_historical_candle_data(
                trading_symbol=symbol,
                exchange=exchange,
                segment=segment,
                start_time=start_time,
                end_time=end_time,
                interval_in_minutes=interval_minutes
            )
            
            if not response or 'candles' not in response:
                logger.error(f"No data returned for {symbol}")
                return pl.DataFrame()
            
            candles = response['candles']
            if not candles:
                logger.warning(f"Empty candles for {symbol}")
                return pl.DataFrame()
            
            # Convert to structured format
            data = []
            for candle in candles:
                # candle format: [epoch_ts, open, high, low, close, volume]
                ts_epoch = candle[0]
                dt = datetime.fromtimestamp(ts_epoch, tz=timezone.utc)
                # Convert to IST (UTC+5:30)
                ist_dt = dt.astimezone(timezone(timedelta(hours=5, minutes=30)))
                
                data.append({
                    'ts': ist_dt,
                    'symbol': symbol,
                    'o': float(candle[1]),
                    'h': float(candle[2]),
                    'l': float(candle[3]),
                    'c': float(candle[4]),
                    'v': int(candle[5]),
                    'vwap': (float(candle[2]) + float(candle[3]) + float(candle[4])) / 3  # approx VWAP
                })
            
            df = pl.DataFrame(data)
            
            # Sort by timestamp
            df = df.sort('ts')
            
            # Cache the data
            df.write_parquet(cache_path)
            logger.info(f"Cached {len(df)} bars for {symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pl.DataFrame()
    
    def load_trades_log(self, log_path: str) -> pl.DataFrame:
        """Load trades log from CSV/Parquet
        
        Expected columns: ts_open, ts_close, symbol, side, qty, entry_px, exit_px, tp, sl, base_score, reason, notes
        """
        path = Path(log_path)
        if not path.exists():
            logger.warning(f"Trades log not found: {log_path}")
            return pl.DataFrame()
        
        try:
            if path.suffix == '.csv':
                df = pl.read_csv(log_path)
            elif path.suffix == '.parquet':
                df = pl.read_parquet(log_path)
            else:
                logger.error(f"Unsupported file format: {path.suffix}")
                return pl.DataFrame()
            
            # Parse timestamps if they're strings
            if df['ts_open'].dtype == pl.Utf8:
                df = df.with_columns([
                    pl.col('ts_open').str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"),
                    pl.col('ts_close').str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
                ])
            
            logger.info(f"Loaded {len(df)} trades from {log_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading trades log: {e}")
            return pl.DataFrame()
    
    def resample_bars(self, df: pl.DataFrame, new_interval_minutes: int, base_interval_minutes: int = 5) -> pl.DataFrame:
        """Resample bars to higher timeframe
        
        Args:
            df: Polars DataFrame with OHLCV data
            new_interval_minutes: Target interval (e.g., 15, 30)
            base_interval_minutes: Current interval (e.g., 5)
            
        Returns:
            Resampled DataFrame
        """
        if new_interval_minutes == base_interval_minutes:
            return df
        
        try:
            # Group by time window and aggregate
            resampled = df.sort('ts').group_by_dynamic(
                'ts',
                every=f"{new_interval_minutes}m"
            ).agg([
                pl.col('symbol').first(),
                pl.col('o').first(),
                pl.col('h').max(),
                pl.col('l').min(),
                pl.col('c').last(),
                pl.col('v').sum(),
                pl.col('vwap').mean()
            ])
            
            logger.info(f"Resampled from {base_interval_minutes}m to {new_interval_minutes}m: {len(df)} -> {len(resampled)} bars")
            return resampled
            
        except Exception as e:
            logger.error(f"Error resampling bars: {e}")
            return df
