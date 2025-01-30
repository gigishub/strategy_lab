import pandas as pd
import pandas_ta as ta
import numpy as np
import datetime
import time
import requests 
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.propagate = False

class SetUpBotKucoin:
    def __init__(self, symbol, timeframe):
        self.symbol = symbol
        self.timeframe = timeframe

        self.high = None
        self.low = None
        self.close = None
        self.open = None

        self.candle_df = None
        self.raw_candle_data = None
        self.df = None
        
        self.atr_sl_value = None
        self.atr_vol_value = None
        self.ema_trend_value = None

        self.trailing_sl_value = None
        self.signal = None
        self.in_trade = 0

    def get_historic_candles(self, market_type="spot", start_time=None, end_time=None):
        base_url = "https://api.kucoin.com" if market_type.lower() == "spot" else "https://api-futures.kucoin.com"
        url = base_url + "/api/v1/market/candles"
        params = {"type": self.timeframe, "symbol": self.symbol.upper()}
        if start_time:
            params["startAt"] = int(start_time.timestamp())
        if end_time:
            params["endAt"] = int(end_time.timestamp())
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") == "200000":
                return data["data"]
            else:
                raise Exception(f"KuCoin API error: {data}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {str(e)}")
        
    def update_til_now(self,lookback_bars=1500):
        endtime = datetime.datetime.now(datetime.timezone.utc)
        starttime = self.calculate_start_time_in_bars(num_bars=lookback_bars)
        self.raw_candle_data = self.get_historic_candles(start_time=starttime, end_time=endtime)
        return self.transform_candle_data(self.raw_candle_data)


    def transform_candle_data(self, raw_candle_data):
        df = pd.DataFrame(raw_candle_data)

        df[['timestamp', 'open', 'close', 'high', 'low']] = df[[0, 1, 2, 3, 4]]
        df.drop([0, 1, 2, 3, 4, 5, 6], axis=1, inplace=True)

        # Explicitly cast 'timestamp' to numeric type
        df['timestamp'] = pd.to_numeric(df['timestamp'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)

        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        df[['open', 'close', 'high', 'low']] = df[['open', 'close', 'high', 'low']].astype(float)

        # set attribute candle_df
        self.candle_df = df
        self.high = df['high']
        self.low = df['low']
        self.close = df['close']
        self.open = df['open']
        return df
    
    def calculate_start_time_in_bars(self, num_bars: int) -> datetime:
        def convert_timeframe_to_timedelta(timeframe: str) -> datetime.timedelta:
            # Convert timeframe to timedelta
            if timeframe.endswith('min'):
                return datetime.timedelta(minutes=int(timeframe[:-3]))
            elif timeframe.endswith('hour'):
                return datetime.timedelta(hours=int(timeframe[:-4]))
            elif timeframe.endswith('day'):
                return datetime.timedelta(days=int(timeframe[:-3]))
            else:
                raise ValueError("Unsupported timeframe format")
        
        # Convert timeframe to timedelta
        timeframe_delta = convert_timeframe_to_timedelta(self.timeframe)
        
        # Calculate the start time based on the number of bars
        start_time = datetime.datetime.now(datetime.timezone.utc) - (num_bars * timeframe_delta)
        return start_time




    def atr_sl(self, length):
        atr = ta.atr(self.high, self.low, self.close, length)
        self.atr_sl_value = atr
        self.atr_sl_value.name = 'ATR_SL'
        return atr

    def atr_volatility(self, lenght):
        atr = ta.atr(self.high, self.low, self.close, lenght)
        self.atr_vol_value = atr
        self.atr_vol_value.name = 'ATR_VOL'
        return atr

    def ema_trend(self, length):
        ema = ta.ema(self.close, length)
        self.ema_trend_value = ema
        self.ema_trend_value.name = 'EMA_TREND'
        return ema
    
    def update_df(self):
        df = self.candle_df
        df['ATR_SL'] = self.atr_sl_value
        df['ATR_VOL'] = self.atr_vol_value
        df['EMA_TREND'] = self.ema_trend_value
        df['SIGNAL'] = self.signal
        df['in_trade'] = self.in_trade
        df['trail_sl'] = self.trailing_sl_value
        self.df = df
        return df

    def get_signal(self, lookback_high,atr_vol_multiplier):
        indicator_calculated = self.ema_trend_value.notna() & self.atr_vol_value.notna() & self.atr_sl_value.notna()
        is_bullish = (self.close > self.ema_trend_value) & indicator_calculated
        rolling_high = self.high.rolling(lookback_high).max()
        is_bearish_vola = (rolling_high - self.low) > (self.atr_vol_value * atr_vol_multiplier)
        conditions = [is_bullish & ~is_bearish_vola, 
                      ~is_bullish | is_bearish_vola]
        
        choices = ["buy", "caution"]

        self.signal = np.select(conditions, choices, default="no_signal")
        self.df['SIGNAL'] = self.signal

        
    # def cacluclate_trail_sl_val(self):
    #     condition = [(self.df['SIGNAL'] == 'buy'),
    #                  (self.df['SIGNAL'] == 'caution'),]
        
    #     choices = [self.df['low'] - self.df['ATR_SL'],
    #                self.df['low'] - self.df['ATR_SL'] * 0.2]

    #     self.df['trail_sl'] = np.select(condition, choices, default=0)
    


    def update_trail_sl(self,i):
        # For each row after the first
        self.df['tsl_values'] = self.df['trail_sl']

        if self.df['in_trade'].iloc[i] == 1 and pd.notna(self.df['trail_sl'].iloc[i]):
            current_tsl = self.df.loc[self.df.index[i],'trail_sl']
            previous_tsl = self.df.loc[self.df.index[i-1],'trail_sl']
            
            # If current value is smaller than previous, use previous
            if pd.notna(previous_tsl):
                if current_tsl < previous_tsl:
                    self.df.loc[self.df.index[i], 'trail_sl'] = previous_tsl
                    self.df.loc[self.df.index[i], 'update_tsl'] = False
                else:
                    self.df.loc[self.df.index[i], 'update_tsl'] = True
        # else:
        #     self.df.loc[self.df.index[i], 'update_tsl'] = None
    

    def execute_buy_trade(self,i):
        if self.df['SIGNAL'].iloc[i] == 'buy':
            self.df.loc[self.df.index[i], 'in_trade'] = 1
            self.df.loc[self.df.index[i], 'trail_sl'] = self.df['low'].iloc[i] - self.df['ATR_SL'].iloc[i]

    def handle_caution(self,i):
        if self.df['SIGNAL'].iloc[i] == 'caution' and self.df['in_trade'].iloc[i-1] == 1:
            self.df.loc[self.df.index[i], 'trail_sl'] = self.df['low'].iloc[i] - self.df['ATR_SL'].iloc[i] * 0.2
            self.df.loc[self.df.index[i], 'in_trade'] = 1

            

    def check_sl(self, i):
        if self.df['in_trade'].iloc[i] == 1 and self.df['close'].iloc[i] < self.df['trail_sl'].iloc[i]:
            self.df.loc[self.df.index[i], 'in_trade'] = 0
            self.df.loc[self.df.index[i], 'trail_sl'] = np.nan
            self.df.loc[self.df.index[i], 'sl_hit'] = True
        else:
            self.df.loc[self.df.index[i], 'sl_hit'] = False



def test1():
    strategy = SetUpBotKucoin('BTC-USDT', timeframe="1day")
    strategy.update_til_now(lookback_bars=200)
    print(strategy.candle_df)

    strategy.atr_sl(14)
    strategy.atr_volatility(14)
    strategy.ema_trend(20)
    strategy.update_df()
    print(strategy.df)

    # strategy.in_trade = 1#np.random.randint(0,2, size=len(strategy.df))
    # strategy.df['in_trade'] = strategy.in_trade
    # strategy.update_trail_sl()
    strategy.get_signal(lookback_high=10, atr_vol_multiplier=2)

    for i in range(1, len(strategy.df)):
        strategy.execute_buy_trade(i)
        strategy.handle_caution(i)
        strategy.update_trail_sl(i)
        strategy.check_sl(i)

    

    print(strategy.df[['SIGNAL','close', 'trail_sl','update_tsl', 'in_trade']].head(40))
    print(strategy.df[['SIGNAL','close', 'trail_sl','update_tsl', 'in_trade','sl_hit']].tail(40))


    
# usage 
if __name__ == '__main__':
    test1()