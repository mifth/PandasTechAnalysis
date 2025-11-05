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


# RSI function
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
    middle_band = ohlc_close.rolling(window=window, min_periods=window).mean()
    std_dev = ohlc_close.rolling(window=window, min_periods=window).std()
    upper_band = middle_band + (std_dev * num_std_dev)
    lower_band = middle_band - (std_dev * num_std_dev)
    return middle_band, upper_band, lower_band

# Calculate Bollinger Band Width
def calculate_bollinger_band_width(upper_band: pd.Series, lower_band: pd.Series, middle_band: pd.Series) -> pd.Series:
    band_width = (upper_band - lower_band) / middle_band
    return band_width

# Calculate Bollinger Band %B
def calculate_bollinger_percent_b(ohlc_close: pd.Series, upper_band: pd.Series, lower_band: pd.Series) -> pd.Series:
    percent_b = (ohlc_close - lower_band) / (upper_band - lower_band)
    return percent_b


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
    atr = tr.ewm(alpha=1.0/window, adjust=False).mean()
    smoothed_plus_dm = plus_dm.ewm(alpha=1.0/window, adjust=False).mean()
    smoothed_minus_dm = minus_dm.ewm(alpha=1.0/window, adjust=False).mean()

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

    # dx = dx.fillna(0)

    # Calculate Average Directional Index (ADX) - Smoothed DX
    adx = dx.ewm(alpha=1.0/window, adjust=False).mean()

    return adx, plus_di, minus_di


# Calculate CCI (Commodity Channel Index)
def calculate_cci(ohlc_high: pd.Series, ohlc_low: pd.Series, ohlc_close: pd.Series, window=20):
    # Typical Price
    tp = (ohlc_high + ohlc_low + ohlc_close) / 3.0

    # Simple Moving Average of TP
    sma_tp = tp.rolling(window=window, min_periods=window).mean()

    # Mean Absolute Deviation (vectorized)
    # Step 1: Difference from SMA
    diff = (tp - sma_tp).abs()

    # Step 2: Rolling mean of the absolute difference
    mad = diff.rolling(window=window, min_periods=window).mean()

    # CCI calculation (with division by zero protection)
    cci = (tp - tp.rolling(window).mean()) / (0.015 * tp.rolling(window).apply(lambda x: np.mean(np.abs(x - np.mean(x)))))

    return cci


# Calculate Stochastic Oscillator
def calculate_stoch(ohlc_high: pd.Series, ohlc_low: pd.Series, ohlc_close: pd.Series,
    k_window=14, d_window=3):
    """
    Calculate Stochastic Oscillator (%K and %D) in a fully vectorized way.
    
    Parameters:
        ohlc_high : High prices (pd.Series)
        ohlc_low  : Low prices (pd.Series)
        ohlc_close: Close prices (pd.Series)
        k_window  : Lookback period for %K (default 14)
        d_window  : Smoothing period for %D (default 3)
    
    Returns:
        stoch_k : %K line
        stoch_d : %D line (SMA of %K)
    """
    # Lowest low and highest high over the k_window
    lowest_low = ohlc_low.rolling(window=k_window, min_periods=k_window).min()
    highest_high = ohlc_high.rolling(window=k_window, min_periods=k_window).max()

    # %K calculation (vectorized, safe for zero range)
    range_ = (highest_high - lowest_low).replace(0, np.nan)
    stoch_k = (ohlc_close - lowest_low) / range_ * 100

    # %D calculation (SMA of %K)
    stoch_d = stoch_k.rolling(window=d_window, min_periods=d_window).mean()

    return stoch_k, stoch_d


# Calculate Awesome Oscillator (AO)
def calculate_ao(
    ohlc_high: pd.Series,
    ohlc_low: pd.Series,
    fast_period=5,
    slow_period=34
):
    """
    Calculate Awesome Oscillator (AO).

    AO = SMA(fast_period, Median Price) - SMA(slow_period, Median Price)
    Median Price = (High + Low) / 2
    """
    median_price = (ohlc_high + ohlc_low) / 2.0

    sma_fast = median_price.rolling(window=fast_period, min_periods=fast_period).mean()
    sma_slow = median_price.rolling(window=slow_period, min_periods=slow_period).mean()

    ao = sma_fast - sma_slow
    return ao


# Calculate Accelerator Oscillator (AC)
def calculate_ac(
    ohlc_high: pd.Series,
    ohlc_low: pd.Series,
    ao_fast=5,
    ao_slow=34,
    ac_sma=5
):
    """
    Calculate Accelerator Oscillator (AC).

    Steps:
    1. Calculate AO = SMA(ao_fast, Median Price) - SMA(ao_slow, Median Price)
    2. Calculate AC = AO - SMA(ac_sma, AO)
    """
    # Median Price
    median_price = (ohlc_high + ohlc_low) / 2.0

    # Awesome Oscillator
    sma_fast = median_price.rolling(window=ao_fast, min_periods=ao_fast).mean()
    sma_slow = median_price.rolling(window=ao_slow, min_periods=ao_slow).mean()
    ao = sma_fast - sma_slow

    # Accelerator Oscillator
    ao_sma = ao.rolling(window=ac_sma, min_periods=ac_sma).mean()
    ac = ao - ao_sma

    return ac


# Calculate Accelerator Oscillator (AC)
def calculate_ac_with_ao(ao: pd.Series, ac_sma_period=5):
    """
    Calculate Accelerator Oscillator (AC).
    ao is already calculated Awesome Oscillator result.
    
    AC = AO - SMA(ac_sma_period, AO)
    """
    ao_sma = ao.rolling(window=ac_sma_period, min_periods=ac_sma_period).mean()
    ac = ao - ao_sma
    return ac


# Calculate Williams %R
def calculate_williams_r(ohlc_high: pd.Series, ohlc_low: pd.Series, 
                         ohlc_close: pd.Series, window=14):
    highest_high = ohlc_high.rolling(window=window, min_periods=window).max()
    lowest_low = ohlc_low.rolling(window=window, min_periods=window).min()

    # Williams %R calculation (vectorized, safe for zero range)
    range_ = (highest_high - lowest_low).replace(0, np.nan)
    williams_r = (highest_high - ohlc_close) / range_ * -100

    return williams_r

# Calculate On-Balance Volume (OBV)
def calculate_obv(ohlc_close: pd.Series, ohlc_volume: pd.Series):
    direction = np.where(ohlc_close.diff() > 0, 1, np.where(ohlc_close.diff() < 0, -1, 0))
    obv = (direction * ohlc_volume).cumsum().fillna(0) #fillna(0) is a safe approach.
    return obv

# Calculate Money Flow Index (MFI)
def calculate_mfi(ohlc_high: pd.Series, ohlc_low: pd.Series, 
                  ohlc_close: pd.Series, ohlc_volume: pd.Series, window=14):
    tp = (ohlc_high + ohlc_low + ohlc_close) / 3.0
    rmf = tp * ohlc_volume
    
    # Calculate positive and negative money flows
    flow_diff = tp.diff()
    positive_flow = rmf.where(flow_diff > 0, 0)
    negative_flow = rmf.where(flow_diff < 0, 0)
    
    positive_mf = positive_flow.rolling(window=window, min_periods=window).sum()
    negative_mf = negative_flow.rolling(window=window, min_periods=window).sum()
    
    mfr = positive_mf / negative_mf.replace(0, np.nan)
    mfi = 100 - (100.0 / (1 + mfr))
    
    return mfi


# Calculate Chaikin Money Flow (CMF)
def calculate_cmf(ohlc_high: pd.Series, ohlc_low: pd.Series, 
                  ohlc_close: pd.Series, ohlc_volume: pd.Series, window=20):
    # Money Flow Multiplier
    mf_multiplier = (((ohlc_close - ohlc_low) - (ohlc_high - ohlc_close)) 
                     / (ohlc_high - ohlc_low).replace(0, np.nan))

    # Money Flow Volume
    mf_volume = mf_multiplier * ohlc_volume

    # CMF calculation
    cmf = (mf_volume.rolling(window=window, min_periods=window).sum() 
           / ohlc_volume.rolling(window=window, min_periods=window).sum().replace(0, np.nan))

    return cmf

# Calculate Force Index (FI)
def calculate_fi(ohlc_close: pd.Series, ohlc_volume: pd.Series, window=13):
    # Raw Force Index
    fi_raw = ohlc_close.diff() * ohlc_volume

    # Smoothed Force Index using EMA
    fi = fi_raw.ewm(span=window, adjust=False).mean()

    return fi

# Calculate Keltner Channels
def calculate_keltner_channels(ohlc_high: pd.Series, ohlc_low: pd.Series, 
                               ohlc_close: pd.Series, window=20, atr_window=10, multiplier=2):
    # Middle Line (EMA of Close)
    middle_line = ohlc_close.ewm(span=window, adjust=False).mean()

    # Average True Range (ATR)
    atr = calculate_atr(ohlc_high, ohlc_low, ohlc_close, window=atr_window)

    # Upper and Lower Bands
    upper_band = middle_line + (multiplier * atr)
    lower_band = middle_line - (multiplier * atr)

    return middle_line, upper_band, lower_band

