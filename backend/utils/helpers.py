"""Helper Functions"""
import polars as pl
from datetime import datetime, timedelta, timezone
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)

def get_ist_timezone():
    """Get IST timezone (UTC+5:30)"""
    return timezone(timedelta(hours=5, minutes=30))

def format_date_for_api(dt: datetime) -> str:
    """Format datetime for GrowwAPI
    
    Args:
        dt: datetime object
        
    Returns:
        String in format 'YYYY-MM-DD HH:MM:SS'
    """
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def parse_log_line(log_line: str) -> dict:
    """Parse trading log line
    
    Example: '2025-11-06 12:23:44,020 INFO Top scanned (symbol,score,ltp,atr%,vwap_dev,vol): [("NSE-RPOWER", 0.132, 40.94, 0.0015, 0.0024, 139)]'
    
    Returns:
        Dict with parsed fields
    """
    try:
        # Extract timestamp
        timestamp_str = log_line.split(' INFO ')[0].split(',')[0]
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        
        # Extract symbol and score if present
        if 'sell status=' in log_line or 'buy status=' in log_line:
            # Trade execution line
            parts = log_line.split('Worker[')
            if len(parts) > 1:
                symbol = parts[1].split(']')[0]
                return {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'type': 'execution'
                }
        elif 'Top scanned' in log_line:
            # Scanner line with scores
            return {
                'timestamp': timestamp,
                'type': 'scan',
                'raw': log_line
            }
        
        return {'timestamp': timestamp, 'type': 'other', 'raw': log_line}
        
    except Exception as e:
        logger.warning(f"Error parsing log line: {e}")
        return {'type': 'unparsed', 'raw': log_line}

def generate_date_range(
    start_date: str,
    end_date: str,
    freq: str = 'daily'
) -> List[Tuple[str, str]]:
    """Generate date ranges for batch processing
    
    Args:
        start_date: 'YYYY-MM-DD'
        end_date: 'YYYY-MM-DD'
        freq: 'daily', 'weekly', 'monthly'
        
    Returns:
        List of (start, end) date string tuples
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    ranges = []
    current = start_dt
    
    while current <= end_dt:
        if freq == 'daily':
            range_end = current
            ranges.append((current.strftime("%Y-%m-%d"), range_end.strftime("%Y-%m-%d")))
            current += timedelta(days=1)
        elif freq == 'weekly':
            range_end = min(current + timedelta(days=6), end_dt)
            ranges.append((current.strftime("%Y-%m-%d"), range_end.strftime("%Y-%m-%d")))
            current += timedelta(weeks=1)
        elif freq == 'monthly':
            # Move to next month
            if current.month == 12:
                range_end = current.replace(year=current.year + 1, month=1, day=1) - timedelta(days=1)
            else:
                range_end = current.replace(month=current.month + 1, day=1) - timedelta(days=1)
            range_end = min(range_end, end_dt)
            ranges.append((current.strftime("%Y-%m-%d"), range_end.strftime("%Y-%m-%d")))
            # Move to first day of next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1, day=1)
            else:
                current = current.replace(month=current.month + 1, day=1)
    
    return ranges
