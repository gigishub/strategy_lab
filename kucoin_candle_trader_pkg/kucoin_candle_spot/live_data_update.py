import pandas as pd
import logging
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import Optional, List

from .kucoin_fetch_spot import SpotDataFetcher
from .kucoin_websocket import KucoinCandlestickWebSocket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CandleData:
    timestamp: int = None
    open: float = None
    close: float = None
    high: float = None
    low: float = None
    volume: float = None
    turnover: float = None


class CandleUpdate:
    def __init__(self, symbol: str, timeframe: str, bars_lookback: int):
        """
        Initialize CandleUpdate with trading pair and timeframe settings.
        
        Args:
            symbol: Trading pair (e.g., "BTC-USDT")
            timeframe: Candle interval (e.g., "1min")
            bars_lookback: Number of historical bars to maintain
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.bars_lookback = bars_lookback
        self.candle_data = CandleData()
        
        # Initialize components
        self.initialize_data_structures()
        
    def initialize_data_structures(self):
        """Initialize time ranges and data fetching components"""
        self.set_time_range()
        self.initialize_objects_needed()
        self.candle_list = self.fetcher.fetch_all_candles()
        self.df_to_trade = self.fetcher.build_dataframe(self.candle_list)

    def set_time_range(self):
        """Calculate start time and set end time to current time"""
        self.calculate_start_time_with_bars()
        self.end_time_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    def initialize_objects_needed(self):
        """Initialize data fetcher and websocket components"""
        self.fetcher = SpotDataFetcher(
            self.symbol, 
            self.timeframe, 
            self.start_time_str, 
            self.end_time_str
        )
        self.ws = KucoinCandlestickWebSocket(self.symbol, self.timeframe)

    def calculate_start_time_with_bars(self):
        """Calculate lookback period based on timeframe"""
        timeframe_map = {
            'min': ('minutes', lambda x: int(x[:-3])),
            'hour': ('hours', lambda x: int(x[:-4])),
            'day': ('days', lambda x: int(x[:-3]))
        }
        
        for suffix, (unit, parser) in timeframe_map.items():
            if self.timeframe.endswith(suffix):
                value = parser(self.timeframe)
                delta = timedelta(**{unit: value})
                break
        else:
            raise ValueError("Unsupported timeframe")
        
        self.start_time = datetime.now(timezone.utc) - (self.bars_lookback * delta)
        self.start_time = self.start_time.replace(second=0, microsecond=0)
        self.start_time_str = self.start_time.strftime("%Y-%m-%d %H:%M:%S")

    def process_candle_data(self, data: dict) -> Optional[List]:
        """
        Process incoming candle data and update internal state.
        Args:
            data: Raw candle data from websocket
            
        Returns:
            Processed candle data as list if valid, None otherwise
        """
        if not data:
            return None
            
        candle = data['candles']
        # Update internal state with proper type conversion
        self.candle_data.timestamp = int(candle[0])
        self.candle_data.open = float(candle[1])
        self.candle_data.close = float(candle[2])
        self.candle_data.high = float(candle[3])
        self.candle_data.low = float(candle[4])
        self.candle_data.volume = float(candle[5])
        self.candle_data.turnover = float(candle[6])
        
        return [
            self.candle_data.timestamp,
            self.candle_data.open,
            self.candle_data.close,
            self.candle_data.high,
            self.candle_data.low,
            self.candle_data.volume,
            self.candle_data.turnover
        ]

    def start_ws(self):
        """Start websocket connection with error handling"""
        try:
            self.ws.start()
        except Exception as e:
            logger.error(f"Failed to start websocket: {e}")
            raise

    def stop_ws(self):
        """Stop websocket connection gracefully"""
        self.ws.stop()

    def new_candle_update(self, include_vol_and_turnover: bool = False) -> Optional[pd.DataFrame]:
        """
        Process new candle updates and return updated DataFrame.
        
        Args:
            include_vol_and_turnover: Whether to include volume and turnover data
            
        Returns:
            Updated DataFrame if new candle, None otherwise
        """
        try:
            data = self.ws.get_data()       
            candle = self.process_candle_data(data)
            
            if not candle:
                return None

            if candle[0] != self.candle_list[-1][0]:
                logger.info('New candle detected, updating DataFrame')
                
                # Update candle list
                self.candle_list.append(candle)
                if len(self.candle_list) > self.bars_lookback:
                    self.candle_list = self.candle_list[-self.bars_lookback:]

                # Create updated DataFrame
                self.df_to_trade = self.create_dataframe(
                    self.candle_list[-self.bars_lookback:],
                    include_vol_and_turnover
                )
                return self.df_to_trade
            else:
                self.candle_list[-1] = candle
                
        except Exception as e:
            logger.error(f"Error updating candles: {e}")
            logger.info('Attempting restart')
            
        return None

    def create_dataframe(self, candles: list, include_vol_and_turnover: bool) -> pd.DataFrame:
        """
        Create DataFrame from candle data.
        Args:
            candles: List of candle data
            include_vol_and_turnover: Whether to include volume and turnover
        Returns:
            Processed DataFrame
        """
        df = pd.DataFrame(
            candles,
            columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover']
        )
        
        # Convert price columns to float explicitly
        for col in ['open', 'close', 'high', 'low', 'volume', 'turnover']:
            df[col] = df[col].astype(float)
            
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s', utc=True)
        
        if not include_vol_and_turnover:
            df.drop(columns=['volume', 'turnover'], inplace=True)
            
        df.set_index('timestamp', inplace=True)
        return df


    def inter_candle_df(self, include_vol_and_turnover: bool = False) -> Optional[pd.DataFrame]:
        """
        Get DataFrame for partially formed candle.
        
        Args:
            include_vol_and_turnover: Whether to include volume and turnover
            
        Returns:
            DataFrame with current partial candle data
        """
        data = self.ws.get_data()       
        candle = self.process_candle_data(data)
        
        if candle:
            return self.create_dataframe([candle], include_vol_and_turnover)
        return None



def main():
    symbol = "BTC-USDT"
    timeframe = "1day"
    error_count = 0
    max_errors = 5

    try:
        candle_update = CandleUpdate(symbol, timeframe, bars_lookback=500)
        candle_update.start_ws()

        # Get initial snapshot
        snapshot = candle_update.df_to_trade
        logger.info(f'Initial snapshot:\n{snapshot}')

        while True:
            # Check for new completed candles
            new_candle_df = candle_update.new_candle_update()
            if new_candle_df is not None:
                logger.info(f'New candle:\n{new_candle_df}')
                # Trade logic for completed candles here
            
            # Check partial candle data
            inter_candle_df = candle_update.inter_candle_df()
            if inter_candle_df is not None:
                logger.info(f'Partial candle:\n{inter_candle_df}')
                # Trade logic for partial candles here
                


    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
        candle_update.stop_ws()
        
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        error_count += 1
        if error_count > max_errors:
            logger.error('Too many errors, shutting down')
            candle_update.stop_ws()

if __name__ == "__main__":
    main()