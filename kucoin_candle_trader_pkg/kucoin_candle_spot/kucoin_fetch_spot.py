import logging
from datetime import datetime, timezone
import time
import requests
import pandas as pd
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FetchMetrics:
    chunks_fetched: int = 0
    total_candles: int = 0
    last_fetch_time: float = 0

class SpotDataFetcher:
    def __init__(self, symbol: str, timeframe: str, start_time: str, end_time: str):
        """
        Initialize the SpotDataFetcher with symbol, timeframe, start_time, and end_time.
        
        Args:
            symbol: Trading pair (e.g., "BTC-USDT")
            timeframe: Candle interval (e.g., "1min")
            start_time: Start time (format: "%Y-%m-%d %H:%M:%S")
            end_time: End time (format: "%Y-%m-%d %H:%M:%S")
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_time = start_time
        self.end_time = end_time
        self.api_url = "https://api.kucoin.com"
        self.metrics = FetchMetrics()

    def make_api_request(self, params: dict) -> dict:
        """Make API request with error handling"""
        try:
            response = requests.get(
                f"{self.api_url}/api/v1/market/candles", 
                params=params, 
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get("code") == "200000":
                return data
            raise Exception(f"KuCoin API error: {data}")
            
        except Exception as e:
            logger.error(f"API request failed: {e}")
            raise

    def fetch_candles_chunk(self, start_time: str, end_time: str) -> list:
        """
        Fetch a single chunk of candlestick data.
        
        Args:
            start_time: Chunk start time
            end_time: Chunk end time
            
        Returns:
            List of candlestick data
        """
        time.sleep(0.5)  # Rate-limit friendly
        
        params = {
            "type": self.timeframe,
            "symbol": self.symbol.upper()
        }
        
        if start_time:
            params["startAt"] = int(time.mktime(time.strptime(start_time, "%Y-%m-%d %H:%M:%S")))
        if end_time:
            params["endAt"] = int(time.mktime(time.strptime(end_time, "%Y-%m-%d %H:%M:%S")))

        response = self.make_api_request(params)
        self.metrics.chunks_fetched += 1
        self.metrics.last_fetch_time = time.time()
        
        return response["data"]

    def fetch_all_candles(self) -> list:
        """
        Fetch all candlesticks in chunks until start_time is reached.
        
        Returns:
            List of all candlestick data within time range
        """
        chunks = []
        current_end = self.end_time
        start_ts = int(time.mktime(time.strptime(self.start_time, "%Y-%m-%d %H:%M:%S")))
        
        logger.info('Fetching candle data...')
        
        while True:
            chunk = self.fetch_candles_chunk(self.start_time, current_end)
            if not chunk:
                break
                
            earliest_ts = int(chunk[-1][0])
            chunks.extend(chunk)
            
            if earliest_ts <= start_ts:
                break
                
            current_end = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(earliest_ts - 60))

        if not chunks:
            return []

        # Sort by timestamp
        chunks.sort(key=lambda x: x[0])
        
        # Filter to requested time range
        filtered_chunks = [
            c for c in chunks 
            if start_ts <= int(c[0]) <= int(time.mktime(time.strptime(self.end_time, "%Y-%m-%d %H:%M:%S")))
        ]
        
        self.metrics.total_candles = len(filtered_chunks)
        return filtered_chunks

    def build_dataframe(self, candles: list) -> pd.DataFrame:
        """
        Convert raw candle data to DataFrame.
        
        Args:
            candles: List of candlestick data
            
        Returns:
            Pandas DataFrame with candlestick data
        """
        df = pd.DataFrame(
            candles, 
            columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover']
        )
        
        # Convert and set timestamp
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        # Convert price/volume columns to float
        df[['open', 'close', 'high', 'low']] = df[['open', 'close', 'high', 'low']].astype(float)
        
        return df

    def fetch_candles_as_df(self) -> pd.DataFrame:
        """
        High-level method to fetch all candles as DataFrame.
        
        Returns:
            DataFrame with all candlestick data
        """
        candles = self.fetch_all_candles()
        df = self.build_dataframe(candles)
        
        logger.info(
            f"Fetch complete. Chunks: {self.metrics.chunks_fetched}, "
            f"Candles: {self.metrics.total_candles}"
        )
        
        return df

def main():
    symbol = "BTC-USDT"
    timeframe = "1min"
    start_time = "2025-01-08 10:00:00"
    end_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    fetcher = SpotDataFetcher(symbol, timeframe, start_time, end_time)
    
    try:
        # Get as DataFrame
        df = fetcher.fetch_candles_as_df()
        print("\nDataFrame output:")
        print(df)
        
        # Get as list
        candles = fetcher.fetch_all_candles()
        print("\nRaw candles output:")
        print(candles[:2])  # Show first 2 candles
        
    except Exception as e:
        logger.error(f"Error fetching data: {e}")

if __name__ == "__main__":
    main()