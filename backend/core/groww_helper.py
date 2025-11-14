"""GrowwAPI Helper - User-Controlled Data Fetching"""
import polars as pl
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class GrowwHelper:
    """Helper class for GrowwAPI integration
    
    ⚠️ USER CUSTOMIZATION AREA ⚠️
    
    Modify the fetch_historical_data() method to use your own GrowwAPI implementation.
    The function signature must remain the same, but you can change the internal logic.
    """
    
    def __init__(self):
        self.last_fetch_params = {}
    
    def fetch_historical_data(
        self,
        trading_symbol: str,
        start_time: str,
        end_time: str,
        interval_in_minutes: int = 5
    ) -> pl.DataFrame:
        """Fetch historical candle data from GrowwAPI
        
        ⭐⭐⭐ USER: PASTE YOUR GROWWAPI CODE HERE ⭐⭐⭐
        
        This function is intentionally left for YOU to implement with your working GrowwAPI code.
        
        TEMPLATE FOR YOUR IMPLEMENTATION:
        ===============================================
        
        import growwapi as groww
        
        # 1. Initialize your Groww client (if needed)
        groww_client = groww.Groww()
        # OR use your existing client
        
        # 2. Set exchange and segment constants
        exchange = groww.EXCHANGE_NSE
        segment = groww.SEGMENT_CASH
        
        # 3. Map interval to Groww constant
        interval_map = {
            1: groww.CANDLE_INTERVAL_MIN_1,
            2: groww.CANDLE_INTERVAL_MIN_2,
            3: groww.CANDLE_INTERVAL_MIN_3,
            5: groww.CANDLE_INTERVAL_MIN_5,
            10: groww.CANDLE_INTERVAL_MIN_10,
            15: groww.CANDLE_INTERVAL_MIN_15,
            30: groww.CANDLE_INTERVAL_MIN_30,
            60: groww.CANDLE_INTERVAL_HOUR_1,
            240: groww.CANDLE_INTERVAL_HOUR_4,
            1440: groww.CANDLE_INTERVAL_DAY_1
        }
        candle_interval = interval_map.get(interval_in_minutes, interval_in_minutes)
        
        # 4. Call YOUR working GrowwAPI method
        response = groww_client.get_historical_candles(
            exchange=exchange,
            segment=segment,
            groww_symbol=trading_symbol,  # e.g., "NSE-RELIANCE"
            start_time=start_time,        # "YYYY-MM-DD HH:MM:SS"
            end_time=end_time,            # "YYYY-MM-DD HH:MM:SS"
            candle_interval=candle_interval
        )
        
        # 5. Parse response into standard format
        candles = response.get('candles', [])
        
        data = []
        for candle in candles:
            # candle format: [epoch_ts, open, high, low, close, volume]
            ts_epoch = candle[0]
            dt = datetime.fromtimestamp(ts_epoch, tz=timezone.utc)
            ist_dt = dt.astimezone(timezone(timedelta(hours=5, minutes=30)))
            
            data.append({
                'ts': ist_dt,
                'symbol': trading_symbol,
                'o': float(candle[1]),
                'h': float(candle[2]),
                'l': float(candle[3]),
                'c': float(candle[4]),
                'v': int(candle[5]),
                'vwap': (float(candle[2]) + float(candle[3]) + float(candle[4])) / 3
            })
        
        df = pl.DataFrame(data)
        return df
        
        ===============================================
        END OF TEMPLATE
        ===============================================
        
        Args:
            trading_symbol: Symbol with exchange prefix (e.g., "NSE-RELIANCE")
            start_time: Start timestamp "YYYY-MM-DD HH:MM:SS"
            end_time: End timestamp "YYYY-MM-DD HH:MM:SS"
            interval_in_minutes: Candle interval (1, 2, 3, 5, 10, 15, 30, 60, 240, 1440)
            
        Returns:
            Polars DataFrame with columns: [ts, symbol, o, h, l, c, v, vwap]
        """
        
        # Store params for debugging
        self.last_fetch_params = {
            'trading_symbol': trading_symbol,
            'start_time': start_time,
            'end_time': end_time,
            'interval_in_minutes': interval_in_minutes
        }
        
        logger.info(f"📥 Fetching data for {trading_symbol}")
        logger.info(f"   Time: {start_time} to {end_time}")
        logger.info(f"   Interval: {interval_in_minutes} min")
        
        # ⭐⭐⭐ REPLACE THIS SECTION WITH YOUR WORKING CODE ⭐⭐⭐
        
        try:
            # === YOUR GROWWAPI CODE GOES HERE ===
            # Example:
            # import growwapi as groww
            # response = groww_client.get_historical_candles(...)
            # Parse response and return DataFrame
            
            # FOR NOW: Return error message
            raise NotImplementedError(
                "\n\n"
                "=" * 60 + "\n"
                "⚠️  GROWW API INTEGRATION REQUIRED\n"
                "=" * 60 + "\n\n"
                "Please implement the fetch_historical_data() method in:\n"
                "  📁 /app/backend/core/groww_helper.py\n\n"
                "Follow the template provided in the docstring above.\n\n"
                "OR use File Upload option to load CSV/Excel data.\n"
                "=" * 60
            )
            
        except NotImplementedError:
            raise
        except Exception as e:
            logger.error(f"❌ GrowwAPI fetch failed: {e}")
            raise
    
    def get_last_fetch_params(self) -> Dict[str, Any]:
        """Get parameters from last fetch attempt (for debugging)"""
        return self.last_fetch_params
