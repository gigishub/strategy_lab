# python
import pandas as pd
import pandas_ta as ta
import numpy as np
import datetime
import requests
import logging
import ccxt
import json
import math
from decimal import Decimal
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.propagate = False

class SetUpBotKucoin:
    def __init__(self, symbol, timeframe):
        self.symbol = symbol
        self.timeframe = timeframe
        self.df = None




    def update_til_now(self, lookback_bars=1500):
        end_time = datetime.datetime.now(datetime.timezone.utc)
        start_time = self.calculate_start_time_in_bars(lookback_bars)
        raw_data = self.get_historic_candles(start_time=start_time, end_time=end_time)
        return self.transform_candle_data(raw_data)

    def get_historic_candles(self, market_type="spot", start_time=None, end_time=None):
        base_url = "https://api.kucoin.com" if market_type.lower() == "spot" else "https://api-futures.kucoin.com"
        url = base_url + "/api/v1/market/candles"
        params = {"type": self.timeframe, "symbol": self.symbol.upper()}
        if start_time:
            params["startAt"] = int(start_time.timestamp())
        if end_time:
            params["endAt"] = int(end_time.timestamp())
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "200000":
            raise Exception(f"KuCoin API error: {data}")
        return data["data"]
        

    def transform_candle_data(self, raw_data):
        df = pd.DataFrame(raw_data)
        df[['timestamp','open','close','high','low']] = df[[0,1,2,3,4]]
        df.drop(columns=[0,1,2,3,4,5,6], inplace=True)
        df['timestamp'] = pd.to_numeric(df['timestamp'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        df[['open','close','high','low']] = df[['open','close','high','low']].astype(float)
        self.df = df
        return df

    def calculate_start_time_in_bars(self, bars):
        if self.timeframe.endswith('min'):
            delta = datetime.timedelta(minutes=int(self.timeframe[:-3]))
        elif self.timeframe.endswith('hour'):
            delta = datetime.timedelta(hours=int(self.timeframe[:-4]))
        elif self.timeframe.endswith('day'):
            delta = datetime.timedelta(days=int(self.timeframe[:-3]))
        else:
            raise ValueError("Unsupported timeframe")
        return datetime.datetime.now(datetime.timezone.utc) - (bars * delta)



    def calculate_indicators(self, atr_length_sl=14, atr_length_vola=20, ema_trend_length=20,ema_is_bullish_length=10):
        self.df['atr_sl'] = ta.atr(self.df['high'], self.df['low'], self.df['close'], atr_length_sl)
        self.df['atr_vola'] = ta.atr(self.df['high'], self.df['low'], self.df['close'], atr_length_vola)
        self.df['ema_trend'] = ta.ema(self.df['close'], ema_trend_length)
        self.df['ema_is_bullish'] = ta.ema(self.df['close'], ema_is_bullish_length)


    def get_signal(self, lookback_high=7, atr_vol_multiplier=1.5):
        is_ready = self.df['atr_sl'].notna() & self.df['atr_vola'].notna() & self.df['ema_trend'].notna()

        is_bullish = (self.df['close'] > self.df['ema_trend']) & (self.df['close'] > self.df['ema_is_bullish']) & is_ready

        rolling_high = self.df['high'].rolling(lookback_high).max()
        is_bearish_vola = (rolling_high - self.df['low']) > (self.df['atr_vola'] * atr_vol_multiplier)
        
        self.df['r_high'] = rolling_high
        self.df['is_bearish_vola'] = is_bearish_vola
        self.df['is_bullish'] = is_bullish

        self.df['signal'] = np.select(
            [is_bullish & ~is_bearish_vola, is_bullish & (is_bearish_vola| (self.df['close']< self.df['ema_trend']))],
            ['buy','caution'],
            default='no_signal'
        )
        self.df['signal'] = self.df['signal'].shift(1)

        self.df['in_trade'] = 0
        self.df['trail_sl'] = np.nan
        self.df['sl_hit'] = False
        self.df['update_tsl'] = False


        self.df['trail_source'] = self.df['low'].rolling(7).max()




    def handle_signal(self, i):
        if self.df['signal'].iloc[i] == 'buy' and self.df['in_trade'].iloc[i-1] == 1:
            self.df.loc[self.df.index[i], 'in_trade'] = 1
            new_sl = self.df['trail_source'].iloc[i] - self.df['atr_sl'].iloc[i]
            # Only update if new stoploss is higher than previous
            if pd.notna(self.df['trail_sl'].iloc[i-1]):
                new_sl = max(new_sl, self.df['trail_sl'].iloc[i-1])
            self.df.loc[self.df.index[i], 'trail_sl'] = new_sl

        elif self.df['signal'].iloc[i] == 'caution' and self.df['in_trade'].iloc[i-1] == 1:
            self.df.loc[self.df.index[i], 'in_trade'] = 1
            new_sl = self.df['trail_source'].iloc[i] - (self.df['atr_sl'].iloc[i] * 0.2)
            # Only update if new stoploss is higher than previous
            if pd.notna(self.df['trail_sl'].iloc[i-1]):
                new_sl = max(new_sl, self.df['trail_sl'].iloc[i-1])
            self.df.loc[self.df.index[i], 'trail_sl'] = new_sl

    def update_trail_sl(self, i):
        if self.df['in_trade'].iloc[i] == 1:
            # Copy previous stoploss by default
            if pd.notna(self.df['trail_sl'].iloc[i-1]):
                self.df.loc[self.df.index[i], 'trail_sl'] = self.df['trail_sl'].iloc[i-1]
            
            # Calculate potential new stoploss based on signal
            if self.df['signal'].iloc[i] == 'buy':
                new_sl = self.df['trail_source'].iloc[i] - self.df['atr_sl'].iloc[i]
            elif self.df['signal'].iloc[i] == 'caution':
                new_sl = self.df['trail_source'].iloc[i] - (self.df['atr_sl'].iloc[i] * 0.2)
            else:
                new_sl = self.df['trail_sl'].iloc[i]

            # Update only if new stoploss is higher
            if pd.notna(new_sl) and pd.notna(self.df['trail_sl'].iloc[i]):
                self.df.loc[self.df.index[i], 'trail_sl'] = max(new_sl, self.df['trail_sl'].iloc[i])


    def set_in_trade(self, i):
        if self.df['signal'].iloc[i] == 'buy' and self.df['in_trade'].iloc[i-1] == 0:
            self.df.loc[self.df.index[i], 'in_trade'] = 1
            self.df.loc[self.df.index[i], 'trail_sl'] = self.df['trail_source'].iloc[i] - self.df['atr_sl'].iloc[i]


    def check_sl(self, i):
        if (self.df['in_trade'].iloc[i-1] == 1 and 
                pd.notna(self.df['trail_sl'].iloc[i-1]) and 
                self.df['close'].iloc[i-1] < self.df['trail_sl'].iloc[i-1]):
            self.df.loc[self.df.index[i], 'in_trade'] = 0
            self.df.loc[self.df.index[i], 'trail_sl'] = np.nan
            self.df.loc[self.df.index[i], 'sl_hit'] = True
        elif self.df['in_trade'].iloc[i-1] == 1:
            self.df.loc[self.df.index[i], 'in_trade'] = 1


    def initialize_kucoin_connection_old(self,config_path: str = '/Users/andre/Documents/Python/trading_clone_singapore_DO/trading_singapore_digitalocean/kucoin_dir/config_api.json') -> ccxt.kucoin:
        """
        Initialize the connection to KuCoin exchange using ccxt.
        
        :param config_path: Path to the JSON file containing API credentials. Defaults to the provided path.
        :return: Initialized ccxt.kucoin exchange object.
        """
        with open(config_path, 'r') as file:
            api_creds = json.load(file)
            
        # Create the exchange object
        exchange = ccxt.kucoin({
            'apiKey': api_creds['api_key'],
            'secret': api_creds['api_secret'],
            'password': api_creds['api_passphrase'],
        })
        
        return exchange
    

    def initialize_kucoin_connection(self) -> ccxt.kucoin:

        # Create the exchange object
        exchange = ccxt.kucoin({
            "apiKey": "66ba4dd1d3e67a000108330f",
            "secret":"c5465410-7eca-43a4-86ac-c63c0e9b8da5",
            "password":"buhyabuhna" 

        })
        
        return exchange
    


        
    def calculate_order_size_buy(self,exchange, symbol: str) -> float:


        # if free == total 50% of balance is used and if free != total 100% of balance is used
        # Fetch the balance
        balance = exchange.fetch_balance()

        usdt_balance = balance['USDT']['free']
        logger.info(f'USDT Total balance: {usdt_balance}')

        basecoin = symbol.split('-')[0]

        basecoin_balance = Decimal(str(balance[basecoin]['free']))
        logger.info(f'{basecoin} balance: {basecoin_balance}')

        # Get the precision of the asset for rounding
        market = exchange.market(symbol)

        precision = market['precision']['amount']
        max_decimals = abs(Decimal(str(precision)).as_tuple().exponent)

        # Get the current price of the asset
        symbol_price = exchange.fetch_ticker(symbol)['last']
        logger.info(f'Current price of {symbol}: {symbol_price}')
        
        # check if basecoin balance is less than 3 times the minimum order size
        # to determine if 
        if basecoin == 'SOL':
            symbol_to_check = 'BTC-USDT'
            btc_market = exchange.market(symbol_to_check)
            # Extract the minimum order size to detremine if BTC is in trade
            min_order_size = Decimal(str(btc_market['limits']['amount']['min']))
            btc_balance = Decimal(str(balance['BTC']['free']))


            if btc_balance <= (min_order_size):
                # if no trade use 50% of balance
                percent_from_bal = 0.5 
                logger.info('BTC is not in trade use 50% of balance to buy SOL if trade will be entered')
            else:
                # use 100% of balance if BTC is in trade
                percent_from_bal = 1
                logger.info('BTC is in trade use 100% of balance to buy SOL if trade will be entered')


        elif basecoin == 'BTC':
            symbol_to_check = 'SOL-USDT'
            sol_market = exchange.market(symbol_to_check)
            # Extract the minimum order size to detremine if SOL is in trade
            min_order_size = Decimal(str(sol_market['limits']['amount']['min']))
            sol_balance = Decimal(str(balance['SOL']['free']))
            
            # if SOL balance is less than 3 times the minimum SOL is NOT in trade
            if sol_balance <= (min_order_size ):
                # if no trade use 50% of balance
                percent_from_bal = 0.5 
                logger.info('SOL is not in trade use 50% of balance to buy BTC if trade will be entered')

            # if SOL balance is more than 3 times the minimum SOL is in trade
            else:
                # use 100% of balance if SOL is in trade
                percent_from_bal = 1
                logger.info('SOL is in trade use 100% of balance to buy BTC if trade will be entered')


        usdt_to_use = usdt_balance * percent_from_bal
        logger.info(f'USDT to use for {symbol}: {usdt_to_use}')

        # Calculate the order size
        order_size = usdt_to_use / symbol_price

        # Round down the order size to the maximum number of decimal places allowed by the exchange
        factor = 10 ** max_decimals
        rounded_order_size = math.floor(order_size * factor) / factor

        return rounded_order_size
    


    def calculate_order_size_sell(self,exchange, symbol: str, percent_from_bal: float) -> float:
            # Fetch the balance
            balance = exchange.fetch_balance()
            coin_balance = balance[symbol.split('-')[0]]['free']

            # print(f'{symbol.split("-")[0]} balance: {coin_balance}')
    
            # Calculate percent of balance to use
            order_size = coin_balance * percent_from_bal

            # Get the precision of the asset for rounding
            market = exchange.market(symbol)
            precision = market['precision']['amount']
            max_decimals = abs(Decimal(str(precision)).as_tuple().exponent)
    
            # Round down the order size to the maximum number of decimal places allowed by the exchange
            factor = 10 ** max_decimals
            rounded_order_size = math.floor(order_size * factor) / factor
    
            return rounded_order_size


    def wait_for_candle_completion(self):
        """
        Waits until the current candle period is complete before proceeding.
        This method blocks until a complete candle is available.
        """
        counter = 0
        while True:
            try:
                if counter > 10:
                    break
                self.update_til_now(lookback_bars=700)
                # Get current time and floor it to the current period
                now = datetime.datetime.now(datetime.timezone.utc)
                
                # Determine timeframe in minutes
                if self.timeframe.endswith('min'):
                    timeframe_minutes = int(self.timeframe[:-3])
                elif self.timeframe.endswith('hour'):
                    timeframe_minutes = int(self.timeframe[:-4]) * 60
                elif self.timeframe.endswith('day'):
                    timeframe_minutes = int(self.timeframe[:-3]) * 1440
                else:
                    raise ValueError("Unsupported timeframe")

                # Floor current time to the timeframe period
                current_period = now.replace(
                    minute=(now.minute // timeframe_minutes) * timeframe_minutes,
                    second=0,
                    microsecond=0
                )

                # Get the timestamp of the last candle in the dataset
                last_candle_time = self.df.index[-1]

                # Check if the last candle is complete
                if last_candle_time == current_period:
                    logger.info("Candle is complete. Proceeding with strategy...")
                    break
                else:
                    logger.info(f"Waiting for candle completion... Current: {now}, Last Candle: {last_candle_time}")
                    time.sleep(10)
                counter += 1
            except Exception as e:
                logger.error(f"Error while checking candle completion: {e}")
                counter += 1
                time.sleep(10)




    def trade_execution(self):

        # test_amount = 0.00001
        
        exchange = self.initialize_kucoin_connection()
        buy_amount = self.calculate_order_size_buy(exchange, self.symbol)
        sell_amount = self.calculate_order_size_sell(exchange, self.symbol, 1)
        
        
        #minimum Amount for testing 
        market = exchange.market(self.symbol)
        # Extract the minimum order size to detremine if SOL is in trade
        buy_amount = Decimal(str(market['limits']['amount']['min']))
        


        # Last candle was just received
        logger.info(f"\n{self.df[['signal', 'open', 'close', 'trail_sl', 'sl_hit', 'in_trade']].tail(15)}")


        if self.df['sl_hit'].iloc[-1]:
            try:
                # Selling logic
                sell_order = exchange.create_market_sell_order(self.symbol, sell_amount)
                logger.info("Trade was stopped out.")
                logger.info(sell_order)
            except Exception as e:
                logger.error(f'Failed to sel: {e}')

        elif self.df['signal'].iloc[-1] == 'buy' and self.df['in_trade'].iloc[-2] == 0:
            # Buying logic
            try:
                buy_order = exchange.create_market_buy_order(self.symbol, buy_amount)
                logger.info("Trade was entered.")
                logger.info(buy_order)
            except Exception as e:
                logger.error(f'Failed to buy: {e}')

        elif self.df['in_trade'].iloc[-1] == 1:
            logger.info("Trade is still active No action.")
        else:
            logger.info("No action.")


        logger.info ('=========================================================================================\n')

            
            
def trade_SOL():
    # strategy parameter 
    symbol = 'SOL-USDT'
    timeframe = '1day'

    # indicator parameters
    atr_length_sl = 5
    atr_length_vola = 5
    ema_trend_length = 240
    ema_is_bullish_length = 10

    # parameters to calculate signal
    lookback_high = 7 # the number of bars to look back to calculate the rolling high for valatility calculation
    atr_vol_multiplier = 1.6 # the multiplier to determine if the volatility is high



    strategy = SetUpBotKucoin(symbol,timeframe)
    # strategy.update_til_now(lookback_bars=700)

    strategy.wait_for_candle_completion()
    logger.info(f"Check {symbol} for trade signal")

    strategy.calculate_indicators(atr_length_sl, atr_length_vola, ema_trend_length,ema_is_bullish_length)
    strategy.get_signal(lookback_high, atr_vol_multiplier)

    for i in range(1, len(strategy.df)):
        # First check if we got stopped out
        strategy.check_sl(i)
        
        # Only proceed with trade management if we're not stopped out
        if not strategy.df['sl_hit'].iloc[i]:
            strategy.handle_signal(i)
            strategy.update_trail_sl(i)
            strategy.set_in_trade(i)


    df = strategy.df

    strategy.trade_execution()


        
            
def trade_BTC():
    # strategy parameter 
    symbol = 'BTC-USDT'
    timeframe = '1day'

    # indicator parameters
    atr_length_sl = 5
    atr_length_vola = 5
    ema_trend_length = 240
    ema_is_bullish_length = 10

    # parameters to calculate signal
    lookback_high = 7 # the number of bars to look back to calculate the rolling high for valatility calculation
    atr_vol_multiplier = 1.6 # the multiplier to determine if the volatility is high
    

    strategy = SetUpBotKucoin(symbol,timeframe)
    # strategy.update_til_now(lookback_bars=700)

    strategy.wait_for_candle_completion()
    logger.info(f"Check {symbol} for trade signal")

    strategy.calculate_indicators(atr_length_sl, atr_length_vola, ema_trend_length,ema_is_bullish_length)
    strategy.get_signal(lookback_high, atr_vol_multiplier)

    for i in range(1, len(strategy.df)):
        # First check if we got stopped out
        strategy.check_sl(i)
        
        # Only proceed with trade management if we're not stopped out
        if not strategy.df['sl_hit'].iloc[i]:
            strategy.handle_signal(i)
            strategy.update_trail_sl(i)
            strategy.set_in_trade(i)


    df = strategy.df

    strategy.trade_execution()



# if __name__ == '__main__':
#     runtime = 3 # minutes for the trading to run
#     start_time = datetime.datetime.now()
#     while start_time + datetime.timedelta(minutes=runtime) > datetime.datetime.now():
#         trade_SOL()
#         trade_BTC()
#         logger.info('Trade check completed for today\n')
#         time.sleep(60)
#     logger.info('testing runtime completed')

if __name__ == '__main__':

    trade_SOL()
    trade_BTC()
    logger.info('Trade check completed for today\n')


