import numpy as np
import pandas as pd


# MA/SMA (Simple Moving Average) function.
# It also can be used for a volume.
def calculate_sma(ohlc_close: pd.Series, window=7):
    return ohlc_close.rolling(window=window).mean()


# EMA (Exponential Moving Average) function.
# It also can be used for a volume.
def calculate_ema(ohlc_close: pd.Series, fast_period=7):
    return ohlc_close.ewm(span=fast_period, adjust=False).mean()


# (Your existing RSI functions - keeping standard version)
def calculate_rsi_standard(ohlc_close: pd.Series, window=14):
    delta = ohlc_close.diff()
    gain = delta.where(delta > 0, 0).rename("gain")
    loss = -delta.where(delta < 0, 0).rename("loss")

    avg_gain = gain.ewm(com=window - 1, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, adjust=False, min_periods=window).mean()

    epsilon = 1e-10
    rs = avg_gain / (avg_loss + epsilon)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # rsi[:window-1] = np.nan # Handled by min_periods in ewm
    return rsi


# Calculate Stochastic RSI. 
# calculated_rsi_window can be None.
# Note that only k_line and d_line outputs are used by graphs mostly.
def calculate_stoch_rsi(ohlc_close: pd.Series, calculated_rsi_window: pd.Series = None, 
                        window=14, smooth_k=3, smooth_d=3) -> tuple:
    # Calculate standard RSI
    rsi = None
    if calculated_rsi_window is None:
        rsi = calculate_rsi_standard(ohlc_close, window=window)
    else:
        rsi = calculated_rsi_window

    # Calculate Stochastic RSI
    stoch_rsi = (rsi - rsi.rolling(window=window).min()) / (rsi.rolling(window=window).max() - rsi.rolling(window=window).min() + 1e-10)

    # Smooth %K line
    k_line = stoch_rsi.rolling(window=smooth_k).mean()

    # Smooth %D line (signal line)
    d_line = k_line.rolling(window=smooth_d).mean()

    return stoch_rsi, k_line, d_line


# Calculate MACD
def calculate_macd(ohlc_close: pd.Series, fast_period=12, slow_period=26, signal_period=9) -> tuple:
    ema_fast = ohlc_close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = ohlc_close.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# Calculate Bollinger Bands
def calculate_bollinger_bands(ohlc_close: pd.Series, window=20, num_std_dev=2) -> tuple:
    middle_band = ohlc_close.rolling(window=window).mean()
    std_dev = ohlc_close.rolling(window=window).std()
    upper_band = middle_band + (std_dev * num_std_dev)
    lower_band = middle_band - (std_dev * num_std_dev)
    return middle_band, upper_band, lower_band


# Calculate ATR
def calculate_atr(ohlc_high: pd.Series, ohlc_low: pd.Series, ohlc_close: pd.Series, window=14):
    high_low = ohlc_high - ohlc_low
    high_close_prev = np.abs(ohlc_high - ohlc_close.shift(1))
    low_close_prev = np.abs(ohlc_low - ohlc_close.shift(1))

    # Calculate True Range (TR)
    tr = pd.DataFrame({'hl': high_low, 'hc': high_close_prev, 'lc': low_close_prev}).max(axis=1)

    # Calculate Average True Range (ATR) using Wilder's smoothing (EWMA with alpha = 1/window)
    atr = tr.ewm(com=(window - 1), adjust=False, min_periods=window).mean()
    return atr


# Calculate ADX (Average Directional Index)
def calculate_adx(ohlc_high: pd.Series, ohlc_low: pd.Series, ohlc_close: pd.Series, window=14,
                  map_to_one: bool = False):
    # Calculate raw price movements
    up_move = ohlc_high.diff()
    down_move = ohlc_low.shift(1) - ohlc_low

    # Calculate +DM and -DM
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    plus_dm = pd.Series(plus_dm, index=ohlc_high.index)
    minus_dm = pd.Series(minus_dm, index=ohlc_high.index)

    # Calculate True Range (TR)
    tr1 = ohlc_high - ohlc_low
    tr2 = np.abs(ohlc_high - ohlc_close.shift(1))
    tr3 = np.abs(ohlc_low - ohlc_close.shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)

    # Smooth the values using Welles Wilder's approximation (EWMA with alpha=1/window)
    atr = tr.ewm(alpha=1/window, adjust=False).mean()
    smoothed_plus_dm = plus_dm.ewm(alpha=1/window, adjust=False).mean()
    smoothed_minus_dm = minus_dm.ewm(alpha=1/window, adjust=False).mean()

    # Calculate Directional Indicators (+DI and -DI)
    plus_di = (smoothed_plus_dm / atr.replace(0, np.nan))
    minus_di = (smoothed_minus_dm / atr.replace(0, np.nan))
    if not map_to_one:
        plus_di *= 100
        minus_di *= 100

    # Calculate Directional Movement Index (DX)
    dx_denominator = (plus_di + minus_di).replace(0, np.nan)
    dx = (np.abs(plus_di - minus_di) / dx_denominator)
    if not map_to_one:
        dx *= 100
    dx = dx.fillna(0)

    # Calculate Average Directional Index (ADX) - Smoothed DX
    adx = dx.ewm(alpha=1/window, adjust=False).mean()

    return adx, plus_di, minus_di
