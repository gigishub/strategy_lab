import vectorbt as vbt
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import pandas_ta as ta
from numba import njit
import gc
import json
import os
from pathlib import Path
from vectorbt.portfolio.enums import SizeType, Direction
from kucoin_candle_spot import SpotDataFetcher

def run_single_pair_analysis(symbol, timeframe, start_time, end_time, params):
    """
    Run analysis for a single trading pair and save results to disk
    """
    print(f"Processing {symbol}...")
    
    # Fetch data
    fetcher = SpotDataFetcher(symbol, timeframe, start_time, end_time)
    df = fetcher.fetch_candles_as_df()
    
    # Calculate trading signals
    indicator = vbt.IndicatorFactory(
        class_name='CustomTradingStrategy',
        short_name='custom_strategy',
        input_names=['Close', 'High', 'Low'],
        param_names=['stretch', 'long_term_ma_len', 'short_term_ma_len', 'adx_len', 'atr_len'],
        output_names=['limit_order_price', 'long_term_ma', 'short_term_ma', 'atr', 'adx']
    ).from_apply_func(custom_trading_strategy)
    
    # Run strategy
    result = indicator.run(
        df['close'], df['high'], df['low'],
        **params,
        param_product=True
    )
    
    # Get signals
    df_indicator_signals = result.limit_order_price
    
    # Prepare close price data
    def repeat_series(series, target_shape):
        return np.tile(series.values.reshape(-1, 1), target_shape[1])
    
    close = pd.DataFrame(
        repeat_series(df['close'], df_indicator_signals.shape),
        index=df.index,
        columns=df_indicator_signals.columns
    )
    
    # Portfolio calculation
    entry_price = np.full(close.shape[0], np.nan)
    pf = vbt.Portfolio.from_order_func(
        close,
        order_func_nb,
        df['high'].to_numpy().flatten(),
        df['low'].to_numpy().flatten(),
        df['open'].to_numpy().flatten(),
        df_indicator_signals.to_numpy(),
        result.short_term_ma.to_numpy().flatten(),
        entry_price,
        init_cash=500
    )
    
    # Calculate key metrics
    metrics = {
        'symbol': symbol,
        'total_return': pf.total_return().to_dict(),
        'max_drawdown': pf.max_drawdown().to_dict(),
        'sharpe_ratio': pf.sharpe_ratio().to_dict(),
        # 'win_rate': pf.win_rate().to_dict(),
        # 'profit_factor': pf.profit_factor().to_dict()
    }
    
    # Save results
    save_results(metrics, symbol)
    
    # Clear memory
    del df, indicator, result, df_indicator_signals, close, pf
    gc.collect()
    
    return True

def custom_trading_strategy(close, high, low, stretch=0.5, 
                          long_term_ma_len=200, short_term_ma_len=20, 
                          adx_len=5, atr_len=3):
    """
    Trading strategy calculation
    """
    atr = vbt.IndicatorFactory.from_talib('ATR').run(high, low, close, timeperiod=atr_len).real.to_numpy()
    adx = vbt.IndicatorFactory.from_talib('ADX').run(high, low, close, timeperiod=adx_len).real.to_numpy()
    long_term_ma = vbt.IndicatorFactory.from_talib('EMA').run(close, timeperiod=long_term_ma_len).real.to_numpy()
    short_term_ma = vbt.IndicatorFactory.from_talib('EMA').run(close, timeperiod=short_term_ma_len).real.to_numpy()
    
    closing_range = (close - low) / (high - low)
    is_uptrend = close > long_term_ma
    is_volatile = adx > 30
    long_limit = low - (atr * stretch)
    long_trigger = closing_range < 0.3
    long_setup = long_trigger & is_uptrend & is_volatile
    limit_order_price = np.where(long_setup, long_limit, np.nan)
    
    return limit_order_price, long_term_ma, short_term_ma, atr, adx

@njit
def order_func_nb(c, high, low, open_, entries, ma_short, entry_price):
    """
    Order function for portfolio calculation
    """
    close_price = c.close[c.i, c.col]
    close_minus_1bar = c.close[c.i-1, c.col]
    
    if c.position_now > 0:
        if (close_minus_1bar <= ma_short[c.i-1]) or (close_price > entry_price[c.i]):
            return vbt.portfolio.nb.order_nb(
                size=-np.inf,
                price=open_[c.i],
                size_type=SizeType.Amount,
                direction=Direction.LongOnly,
                fees=0.001,
                slippage=0.002)
                
    elif (c.position_now == 0) and (c.i != 0):
        if (entries[c.i-1,c.col] > 0) and (low[c.i] < entries[c.i-1,c.col]):
            entry_price[:] = np.nan
            entry_price[:] = entries[c.i-1,c.col]
            
            return vbt.portfolio.nb.order_nb(
                size=1,
                price=entry_price[c.i],
                size_type=SizeType.Percent,
                direction=Direction.LongOnly,
                fees=0.001,
                slippage=0.002,
                allow_partial=False,
                raise_reject=True
            )
    
    return vbt.portfolio.enums.NoOrder

def save_results(metrics, symbol):
    """
    Save results to JSON file
    """
    # Create results directory if it doesn't exist
    results_dir = Path('trading_results')
    results_dir.mkdir(exist_ok=True)
    
    # Save metrics to JSON file
    filename = results_dir / f"{symbol.replace('/', '_')}_results.json"
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)

def load_and_compare_results():
    """
    Load all results and create comparison
    """
    results_dir = Path('trading_results')
    all_results = []
    
    for file in results_dir.glob('*_results.json'):
        with open(file, 'r') as f:
            results = json.load(f)
            all_results.append(results)
    
    # Create comparison DataFrame
    comparison = pd.DataFrame([
        {
            'Symbol': r['symbol'],
            'Best Total Return': max(r['total_return'].values()),
            'Worst Drawdown': min(r['max_drawdown'].values()),
            'Best Sharpe': max(r['sharpe_ratio'].values()),
            # 'Win Rate': max(r['win_rate'].values()),
            # 'Profit Factor': max(r['profit_factor'].values())
        }
        for r in all_results
    ])
    
    return comparison.sort_values('Best Total Return', ascending=False)

# Example usage
if __name__ == "__main__":
    # Configuration
    timeframe = "1day"
    start_time = "2024-01-08 10:00:00"
    end_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    
    params = {
        'stretch': 0.5,
        'long_term_ma_len': [100, 200],
        'short_term_ma_len': 20,
        'adx_len': [5, 6],
        'atr_len': 3
    }
    
    # List of symbols to analyze
    symbols = ["ETH-USDT", "BTC-USDT", "SOL-USDT"]
    
    # Process each symbol separately
    for symbol in symbols:
        try:
            run_single_pair_analysis(symbol, timeframe, start_time, end_time, params)
            print(f"Completed analysis for {symbol}")
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
    
    # Compare results
    comparison = load_and_compare_results()
    print("\nPerformance Comparison:")
    print(comparison)