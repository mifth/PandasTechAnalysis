# Calculate OHLC candles features for machine learning

import pandas as pd


# Candle body size relative to high-low range
def candle_weakness(ohlc_high: pd.Series, ohlc_low: pd.Series, 
                    ohlc_open: pd.Series, ohlc_close: pd.Series) -> pd.Series:
    candle_weakness = ((ohlc_open - ohlc_close).abs() / (ohlc_high - ohlc_low))
    return candle_weakness

# Is the candle a green candle (close > open)
def is_green_candle(ohlc_open: pd.Series, ohlc_close: pd.Series) -> pd.Series:
    return (ohlc_close > ohlc_open).astype(int)

# tallness of current candle vs previous candle
def candle_taller_than_prev_hl(ohlc_high: pd.Series, ohlc_low: pd.Series) -> pd.Series:
    prev = (ohlc_high - ohlc_low).shift(1)
    taller_pct = ((ohlc_high - ohlc_low) / prev) - 1
    return taller_pct

# tallness of current candle vs previous candle body (open-close)
def candle_taller_than_prev_oc(ohlc_open: pd.Series, ohlc_close: pd.Series) -> pd.Series:
    prev = (ohlc_close - ohlc_open).shift(1).abs()
    taller_pct = ((ohlc_close - ohlc_open).abs() / prev) - 1
    return taller_pct

# Center point of the candle (average of high and low)
def candle_center_point(ohlc_high: pd.Series, ohlc_low: pd.Series) -> pd.Series:
    return (ohlc_high + ohlc_low) / 2.0

# Is the close above the center point of the candle
def is_close_above_center(ohlc_high: pd.Series, ohlc_low: pd.Series, ohlc_close: pd.Series) -> pd.Series:
    center = candle_center_point(ohlc_high, ohlc_low)
    return (ohlc_close > center).astype(int)

# Is the open above the center point of the candle
def is_open_above_center(ohlc_high: pd.Series, ohlc_low: pd.Series, ohlc_open: pd.Series) -> pd.Series:
    center = candle_center_point(ohlc_high, ohlc_low)
    return (ohlc_open > center).astype(int)

# Rolling min and max pct change from close
def min_rolling_pct(ohlc_low: pd.Series, ohlc_close: pd.Series, window: int) -> pd.Series:
    return (ohlc_low.rolling(window=window).min() / ohlc_close) - 1

# Rolling max pct change from close
def max_rolling_pct(ohlc_high: pd.Series, ohlc_close: pd.Series, window: int) -> pd.Series:
    return (ohlc_high.rolling(window=window).max() / ohlc_close) - 1

# Close price returns over given periods. periods=1 is a previous candle.
def close_returns(ohlc_close: pd.Series, periods: int = 1) -> pd.Series:
    return ohlc_close.pct_change(periods=periods)

# Rolling range (high - low) relative to previous rolling average range
def rolling_range(ohlc_high: pd.Series, ohlc_low: pd.Series, window: int) -> pd.Series:
    range_1 = (ohlc_high - ohlc_low)
    range_2 = range_1.shift(1).rolling(window=window).mean()
    return range_1 / range_2

