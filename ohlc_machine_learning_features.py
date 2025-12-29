# Calculate OHLC candles features for machine learning

import numpy as np
import pandas as pd


# Candle body size relative to high-low range
def candle_weakness(ohlc_high: pd.Series, ohlc_low: pd.Series, 
                    ohlc_open: pd.Series, ohlc_close: pd.Series) -> pd.Series:
    candle_weakness = ((ohlc_open - ohlc_close).abs() / (ohlc_high - ohlc_low))
    candle_weakness = candle_weakness.replace([np.inf, -np.inf, np.nan], 0)
    return candle_weakness

# Is the candle a green candle (close > open)
def is_green_candle(ohlc_open: pd.Series, ohlc_close: pd.Series) -> pd.Series:
    return (ohlc_close > ohlc_open).astype(int)

# tallness of current candle vs previous candle
def candle_taller_than_prev_hl(ohlc_high: pd.Series, ohlc_low: pd.Series) -> pd.Series:
    prev = (ohlc_high - ohlc_low).shift(1)
    taller_pct = ((ohlc_high - ohlc_low) / prev) - 1
    taller_pct = taller_pct.replace([np.inf, -np.inf, np.nan], 0)
    return taller_pct

# tallness of current candle vs previous candle body (open-close)
def candle_taller_than_prev_oc(ohlc_open: pd.Series, ohlc_close: pd.Series) -> pd.Series:
    prev = (ohlc_close - ohlc_open).shift(1).abs()
    taller_pct = ((ohlc_close - ohlc_open).abs() / prev) - 1
    taller_pct = taller_pct.replace([np.inf, -np.inf, np.nan], 0)
    return taller_pct

# Is the close diff from 0 to 1
def open_or_close_diff(high: pd.Series, low: pd.Series, open_or_close: pd.Series) -> pd.Series:
    size = high - low
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = (open_or_close - low) / size
    pct = pct.replace([np.inf, -np.inf, np.nan], 0)
    return pct

# Rolling min and max pct change from close
def min_rolling_pct(ohlc_low: pd.Series, ohlc_close: pd.Series, window: int) -> pd.Series:
    min_rolling_pct = (ohlc_low.rolling(window=window).min() / ohlc_close) - 1
    min_rolling_pct = min_rolling_pct.replace([np.inf, -np.inf, np.nan], 0)
    return min_rolling_pct

# Rolling max pct change from close
def max_rolling_pct(ohlc_high: pd.Series, ohlc_close: pd.Series, window: int) -> pd.Series:
    max_rolling_pct = (ohlc_high.rolling(window=window).max() / ohlc_close) - 1
    max_rolling_pct = max_rolling_pct.replace([np.inf, -np.inf, np.nan], 0)
    return max_rolling_pct

# Rolling range (high - low) relative to previous rolling average range
def rolling_range(ohlc_high: pd.Series, ohlc_low: pd.Series, window: int) -> pd.Series:
    range_1 = (ohlc_high - ohlc_low)
    range_2 = range_1.shift(1).rolling(window=window).mean()
    final_range = (range_1 / range_2) - 1
    final_range = final_range.replace([np.inf, -np.inf, np.nan], 0)
    return final_range

# Is doji candle
def is_doji(ohlc_open: pd.Series, ohlc_close: pd.Series, ohlc_high: pd.Series, ohlc_low: pd.Series) -> pd.Series:
    # A doji is a candle with a very small body relative to its total range
    body = (ohlc_open - ohlc_close).abs()
    candle_range = ohlc_high - ohlc_low
    # Use a small tolerance value, e.g., 5% of the range
    return (body / candle_range < 0.05).astype(int)

# Is hammer candle
def is_hammer(ohlc_open: pd.Series, ohlc_close: pd.Series, ohlc_high: pd.Series, ohlc_low: pd.Series) -> pd.Series:
    body = (ohlc_close - ohlc_open).abs()
    lower_wick = (ohlc_open - ohlc_low).where(ohlc_close > ohlc_open, (ohlc_close - ohlc_low))
    upper_wick = (ohlc_high - ohlc_close).where(ohlc_close > ohlc_open, (ohlc_high - ohlc_open))

    # A hammer has a small body, a long lower wick (at least twice the body), and a small upper wick.
    return ((lower_wick > 2 * body) & (upper_wick < body) & (body > 0)).astype(int)

# Is marubozu candle
def is_marubozu(ohlc_open: pd.Series, ohlc_close: pd.Series, ohlc_high: pd.Series, ohlc_low: pd.Series) -> pd.Series:
    body = (ohlc_open - ohlc_close).abs()
    candle_range = ohlc_high - ohlc_low

    # A marubozu has a body that is a large percentage of its total range
    return (body / candle_range > 0.8).astype(int)
