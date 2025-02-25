from kucoin_candle_spot import CandleUpdate
import logging


logger = logging.getLogger(__name__)

symbol = "BTC-USDT" 
timeframe = "1day"

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
        


except KeyboardInterrupt:
    logger.info("Shutting down gracefully...")
    candle_update.stop_ws()
    
