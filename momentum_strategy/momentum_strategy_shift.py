import pandas as pd
import pandas_ta as ta
import numpy as np
import datetime
import requests
import logging

class SetUpBotKucoin:
    def __init__(self, symbol, timeframe):
        self.symbol = symbol
        self.timeframe = timeframe
        self.df = None
        self.in_trade = 0

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
        
    def update_til_now(self, lookback_bars=1500):
        end_time = datetime.datetime.now(datetime.timezone.utc)
        start_time = self.calculate_start_time_in_bars(lookback_bars)
        raw_data = self.get_historic_candles(start_time=start_time, end_time=end_time)
        return self.transform_candle_data(raw_data)

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

    def calculate_indicators(self, atr_length_sl=14, atr_length_vola=20, ema_length=20):
        self.df['atr_sl'] = ta.atr(self.df['high'], self.df['low'], self.df['close'], atr_length_sl)
        self.df['atr_vola'] = ta.atr(self.df['high'], self.df['low'], self.df['close'], atr_length_vola)
        self.df['ema_trend'] = ta.ema(self.df['close'], ema_length)

        self.df['close_shift'] = self.df['close'].shift(1)  
        self.df['high_shift'] = self.df['high'].shift(1)
        self.df['low_shift'] = self.df['low'].shift(1)

    def get_signal(self, lookback_high=7, atr_vol_multiplier=1.5):
        is_ready = self.df['atr_sl'].notna() & self.df['atr_vola'].notna() & self.df['ema_trend'].notna()
        is_bullish = (self.df['close'] > self.df['ema_trend']) & is_ready

        rolling_high = self.df['high'].rolling(lookback_high).max()
        is_bearish_vola = (rolling_high - self.df['low']) > (self.df['atr_vola'] * atr_vol_multiplier)

        self.df['r_high'] = rolling_high
        self.df['is_bearish_vola'] = is_bearish_vola
        self.df['is_bullish'] = is_bullish

        self.df['signal'] = np.select(
            [is_bullish & ~is_bearish_vola, ~is_bullish | is_bearish_vola],
            ['buy','caution'],
            default='no_signal'
        )

        self.df['signal_shift'] = self.df['signal'].shift(1)
        self.df['trail_source'] = self.df['low'].rolling(7).max()
        self.df['in_trade'] = 0
        self.df['trail_sl'] = np.nan
        self.df['sl_hit'] = False
        self.df['update_tsl'] = False
        self.df['tsl_buy'] = np.nan
        self.df['tsl_caution'] = np.nan

    def calculate_tsl(self):
        self.df['tsl_buy'] = self.df['trail_source'] - self.df['atr_sl']
        self.df['tsl_caution'] = self.df['trail_source'] - (self.df['atr_sl'] * 0.2)

        self.df['tsl_buy_shift'] = self.df['tsl_buy'].shift(1)
        self.df['tsl_caution_shift'] = self.df['tsl_caution'].shift(1)

    def execute_buy_trade(self):
        self.df['entry'] = np.nan
        buy_mask = (self.df['signal_shift'] == 'buy') & (self.df['in_trade'] == 0)

        # Initialize in_trade column
        self.df['in_trade'] = 0

        # Set in_trade to 1 for buy signals
        self.df.loc[buy_mask, 'in_trade'] = 1
        self.df.loc[buy_mask, 'trail_sl'] = self.df['tsl_buy']

        # Create a temporary column for the propagation
        temp_trade = self.df['in_trade'].copy()
        
        # Propagate in_trade status using shift operations
        for i in range(1, len(self.df)):
            if temp_trade.iloc[i-1] == 1 and not self.df['sl_hit'].iloc[i]:
                temp_trade.iloc[i] = 1
        
        # Update the final in_trade column
        self.df['in_trade'] = temp_trade

        # Calculate in_trade_prev
        self.df['in_trade_prev'] = self.df['in_trade'].shift(1) == 1


    # def execute_buy_trade(self):
    #     self.df['entry'] = np.nan
    #     buy_mask = (self.df['signal_shift'] == 'buy') & (self.df['in_trade'] == 0)

    #     # Set in_trade to 1 for buy signals
    #     self.df.loc[buy_mask, 'in_trade'] = 1
    #     self.df.loc[buy_mask, 'trail_sl'] = self.df['tsl_buy']

    #     # Propagate in_trade status forward
    #     self.df['in_trade'] = self.df['in_trade'].fillna(0)  # Ensure no NaN values
    #     for i in range(1, len(self.df)):
    #         if self.df['in_trade'].iloc[i-1] == 1 and not self.df['sl_hit'].iloc[i]:
    #             self.df['in_trade'].iloc[i] = 1

    #     # Calculate in_trade_prev after propagating the in_trade status
    #     self.df['in_trade_prev'] = self.df['in_trade'].shift(1) == 1

    def update_trail_sl(self):
        signal_buy = (self.df['signal_shift'] == 'buy') & (self.df['in_trade'] == 1)
        signal_caution = (self.df['signal_shift'] == 'caution') & (self.df['in_trade'] == 1)
        
        self.df['in_sig_buy'] = signal_buy
        self.df['in_sig_caution'] = signal_caution
        
        conditions = [signal_buy, signal_caution]
        choices = [self.df['tsl_buy'], self.df['tsl_caution']]
        temp_sl = np.select(conditions, choices, default=np.nan)
        
        prev_sl = self.df['trail_sl'].shift(1)
        self.df['trail_sl'] = np.where(
            self.df['in_trade'] == 1,
            np.where(temp_sl < prev_sl, prev_sl, temp_sl),
            np.nan
        )

    def check_sl(self):
        sl_hit_mask = (self.df['in_trade'] == 1) & (self.df['close_shift'] < self.df['trail_sl'].shift(1))
        self.df['in_trade'] = np.where(sl_hit_mask, 0, self.df['in_trade'])
        self.df['trail_sl'] = np.where(sl_hit_mask, np.nan, self.df['trail_sl'])
        self.df['sl_hit'] = sl_hit_mask

def test1():
    strategy = SetUpBotKucoin('BTC-USDT','1day')
    strategy.update_til_now(lookback_bars=1500)
    strategy.calculate_indicators(atr_length_sl=5, atr_length_vola=5, ema_length=200)
    strategy.get_signal(lookback_high=7, atr_vol_multiplier=1.5)
    strategy.calculate_tsl()
    strategy.execute_buy_trade()
    strategy.update_trail_sl()
    strategy.check_sl()
    print(strategy.df[['signal_shift','close','trail_sl','sl_hit','in_trade_prev','in_trade']].tail(50))

if __name__ == '__main__':
    test1()