"""Data Ingestion Module - GrowwAPI Integration with Duration Limits"""
import polars as pl
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Tuple
import logging
from pathlib import Path
import re

logger = logging.getLogger(__name__)

# Duration limits based on candle intervals (as per Groww backtesting limits)
DURATION_LIMITS = {
    1: 30,   # 1 min -> 30 days
    2: 30,   # 2 min -> 30 days
    3: 30,   # 3 min -> 30 days
    5: 30,   # 5 min -> 30 days
    10: 90,  # 10 min -> 90 days
    15: 90,  # 15 min -> 90 days
    30: 90,  # 30 min -> 90 days
    60: 180, # 1 hour -> 180 days
    240: 180, # 4 hours -> 180 days
    1440: 180, # 1 day -> 180 days
}

class DataIngestion:
    """Handles data fetching from GrowwAPI with duration validation"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def validate_duration(self, start_time: datetime, end_time: datetime, interval_minutes: int) -> Tuple[bool, str]:
        """Validate if duration is within Groww API limits
        
        Returns:
            (is_valid, error_message)
        """
        duration_days = (end_time - start_time).days
        max_days = DURATION_LIMITS.get(interval_minutes, 30)
        
        if duration_days > max_days:
            return False, f"Duration {duration_days} days exceeds limit of {max_days} days for {interval_minutes}min candles"
        
        return True, ""
    
    def get_candle_interval_constant(self, interval_minutes: int):
        """Map interval minutes to Groww constants"""
        # Import here to avoid circular dependency
        try:
            import growwapi as groww
            
            interval_map = {
                1: getattr(groww, 'CANDLE_INTERVAL_MIN_1', 1),
                2: getattr(groww, 'CANDLE_INTERVAL_MIN_2', 2),
                3: getattr(groww, 'CANDLE_INTERVAL_MIN_3', 3),
                5: getattr(groww, 'CANDLE_INTERVAL_MIN_5', 5),
                10: getattr(groww, 'CANDLE_INTERVAL_MIN_10', 10),
                15: getattr(groww, 'CANDLE_INTERVAL_MIN_15', 15),
                30: getattr(groww, 'CANDLE_INTERVAL_MIN_30', 30),
                60: getattr(groww, 'CANDLE_INTERVAL_HOUR_1', 60),
                240: getattr(groww, 'CANDLE_INTERVAL_HOUR_4', 240),
                1440: getattr(groww, 'CANDLE_INTERVAL_DAY_1', 1440),
            }
            
            return interval_map.get(interval_minutes, interval_minutes)
        except:
            return interval_minutes
    
    def fetch_ohlcv_bars(
        self,
        groww_client,
        symbol: str,
        start_time: str,
        end_time: str,
        interval_minutes: int = 5,
        use_cache: bool = True
    ) -> pl.DataFrame:
        """Fetch OHLCV bars from GrowwAPI with proper method and validation
        
        Args:
            groww_client: Authenticated GrowwAPI client
            symbol: Trading symbol with exchange prefix (e.g., 'NSE-RELIANCE')
            start_time: Start timestamp 'YYYY-MM-DD HH:MM:SS'
            end_time: End timestamp 'YYYY-MM-DD HH:MM:SS'
            interval_minutes: Candle interval (1, 2, 3, 5, 10, 15, 30, 60, 240, 1440)
            use_cache: Whether to use cached data
            
        Returns:
            Polars DataFrame with columns: [ts, symbol, o, h, l, c, v, vwap]
        """
        cache_key = f"{symbol}_{interval_minutes}m_{start_time.replace(' ', '_').replace(':', '-')}_{end_time.replace(' ', '_').replace(':', '-')}.parquet"
        cache_path = self.cache_dir / cache_key
        
        # Check cache
        if use_cache and cache_path.exists():
            logger.info(f"✅ Loading cached data for {symbol}")
            return pl.read_parquet(cache_path)
        
        try:
            # Parse dates for validation
            start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
            
            # Validate duration
            is_valid, error_msg = self.validate_duration(start_dt, end_dt, interval_minutes)
            if not is_valid:
                logger.error(f"⚠️ {error_msg}")
                logger.info(f"💡 Adjusting to maximum allowed duration...")
                max_days = DURATION_LIMITS.get(interval_minutes, 30)
                start_dt = end_dt - timedelta(days=max_days)
                start_time = start_dt.strftime("%Y-%m-%d %H:%M:%S")
                logger.info(f"📅 Adjusted range: {start_time} to {end_time}")
            
            logger.info(f"📥 Fetching {symbol} from GrowwAPI ({interval_minutes}min): {start_time} to {end_time}")
            
            # Import growwapi
            import growwapi as groww
            
            # Get candle interval constant
            candle_interval = self.get_candle_interval_constant(interval_minutes)
            
            # Use correct method: get_historical_candles (NOT get_historical_candle_data)
            response = groww_client.get_historical_candles(
                exchange=groww.EXCHANGE_NSE,
                segment=groww.SEGMENT_CASH,
                groww_symbol=symbol,  # Already has NSE- prefix
                start_time=start_time,
                end_time=end_time,
                candle_interval=candle_interval
            )
            
            if not response or 'candles' not in response:
                logger.error(f"❌ No data returned for {symbol}")
                return pl.DataFrame()
            
            candles = response['candles']
            if not candles:
                logger.warning(f"⚠️ Empty candles for {symbol}")
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
            df = df.sort('ts')
            
            # Cache the data
            df.write_parquet(cache_path)
            logger.info(f"✅ Cached {len(df)} bars for {symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error fetching data for {symbol}: {e}")
            return pl.DataFrame()
    
    def parse_trade_log_line(self, log_line: str) -> Optional[Dict]:
        """Parse trade log in your specific format
        
        Example: 'Worker[NSE-MANAPPURAM] starting: entry=272.9 stop_t=266.05 stop_l=266.0 target=279.75 qty=14'
        
        Returns:
            Dict with trade details or None if not a trade line
        """
        try:
            # Match Worker[SYMBOL] pattern
            worker_match = re.search(r'Worker\[([^\]]+)\]', log_line)
            if not worker_match:
                return None
            
            symbol = worker_match.group(1)
            
            # Extract trade details
            entry_match = re.search(r'entry=([\d.]+)', log_line)
            stop_t_match = re.search(r'stop_t=([\d.]+)', log_line)
            stop_l_match = re.search(r'stop_l=([\d.]+)', log_line)
            target_match = re.search(r'target=([\d.]+)', log_line)
            qty_match = re.search(r'qty=([\d]+)', log_line)
            
            # Extract timestamp (at beginning of line)
            ts_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', log_line)
            
            if entry_match:
                return {
                    'timestamp': datetime.strptime(ts_match.group(1), "%Y-%m-%d %H:%M:%S") if ts_match else None,
                    'symbol': symbol,
                    'entry': float(entry_match.group(1)),
                    'stop_trailing': float(stop_t_match.group(1)) if stop_t_match else None,
                    'stop_loss': float(stop_l_match.group(1)) if stop_l_match else None,
                    'target': float(target_match.group(1)) if target_match else None,
                    'qty': int(qty_match.group(1)) if qty_match else None,
                    'type': 'entry'
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Error parsing log line: {e}")
            return None
    
    def parse_scanner_log_line(self, log_line: str) -> List[Dict]:
        """Parse scanner output in your format
        
        Example: '(symbol,score,ltp,atr%,vwap_dev,vol): [("NSE-MANAPPURAM", 2.002, 272.9, 0.0002, 0.0001, 3000)]'
        
        Returns:
            List of scanned stock dicts
        """
        try:
            # Find the list portion
            match = re.search(r'\[\(([^\]]+)\)\]', log_line)
            if not match:
                return []
            
            # Parse tuples
            stocks = []
            tuple_pattern = r"\('([^']+)',\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d]+)\)"
            
            for match in re.finditer(tuple_pattern, log_line):
                stocks.append({
                    'symbol': match.group(1),
                    'score': float(match.group(2)),
                    'ltp': float(match.group(3)),
                    'atr_pct': float(match.group(4)),
                    'vwap_dev': float(match.group(5)),
                    'volume': int(match.group(6))
                })
            
            return stocks
            
        except Exception as e:
            logger.warning(f"Error parsing scanner line: {e}")
            return []
    
    def load_trade_logs(self, log_file_path: str) -> Tuple[List[Dict], List[Dict]]:
        """Load and parse trade logs from file
        
        Returns:
            (scanner_results, trade_entries)
        """
        scanner_results = []
        trade_entries = []
        
        log_path = Path(log_file_path)
        if not log_path.exists():
            logger.warning(f"⚠️ Log file not found: {log_file_path}")
            return scanner_results, trade_entries
        
        try:
            with open(log_path, 'r') as f:
                for line in f:
                    # Parse scanner results
                    if 'symbol,score,ltp,atr%,vwap_dev,vol' in line:
                        stocks = self.parse_scanner_log_line(line)
                        scanner_results.extend(stocks)
                    
                    # Parse trade entries
                    if 'Worker[' in line and 'starting:' in line:
                        trade = self.parse_trade_log_line(line)
                        if trade:
                            trade_entries.append(trade)
            
            logger.info(f"✅ Parsed {len(scanner_results)} scanner results, {len(trade_entries)} trade entries")
            return scanner_results, trade_entries
            
        except Exception as e:
            logger.error(f"❌ Error loading trade logs: {e}")
            return scanner_results, trade_entries
