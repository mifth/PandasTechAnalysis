# PandasTechAnalysis
This is a Technical Analysis Python library for markets using Pandas/Numpy. It's fast and simple.

<img width="1488" height="875" alt="image" src="https://github.com/user-attachments/assets/cc3797b4-9936-437f-831b-73b19aebf4ac" />


Tech Analysis:
- SMA
- EMA
- MACD
- ATR
- Bollinger Bands
- Bollinger %B
- Bollinger Band Width
- RSI
- Stoch RSI
- ADX
- Commodity Channel Index (CCI)
- Stochastic Oscillator (Stoch)
- Awesome Oscillator (AO)
- Accelerator Oscillator (AC)
- Williams %R
- On-Balance Volume (OBV)
- Money Flow Index (MFI)
- Chaikin Money Flow (CMF)
- Force Index (FI)
- Keltner Channels
- Trend Strength Indicator (TSI)

Dependencies:
- Pandas
- Numpy

How to use:
- Get your candles/values data and convert it to Pandas data.
- Calculate tech analysis with PandasTechAnalysis.
- Visualize with Plotly or Bokeh.

Example:
```
import numpy as np
import pandas as pd
import pandas_tech_analysis as pta

candles_df: pd.DataFrame  # Your retrieved Candles/Values data from a market.
# Say, your candles_df['open'] is candles open values.
# Say, your candles_df['high'] is candles high values.
# Say, your candles_df['low'] is candles low values.
# Say, your candles_df['close'] is candles close values.
# Say, your candles_df['volume'] is volume values.

# Candles Candles EMA
ema_7 = pta.calculate_ema(candles_df['close'], 7)
ema_14 = pta.calculate_ema(candles_df['close'], 14)
ema_28 = pta.calculate_ema(candles_df['close'], 28)

# RSI
rsi_14 = pta.calculate_rsi_standard(candles_df['close'], window=14)
rsi_28 = pta.calculate_rsi_standard(candles_df['close'], window=28)
rsi_42 = pta.calculate_rsi_standard(candles_df['close'], window=42)

# Calculate Volume SMA
volume_sma_5 = pta.calculate_sma(candles_df['volume'], 5)
volume_sma_10 = pta.calculate_sma(candles_df['volume'], 10)

# Stoch RSI. Note only stoc_k and stoch_d values are used mostly.
stoch_rsi, stoc_k, stoch_d = pta.calculate_stoch_rsi(candles_df['close'], rsi_14, window=14)

# MACD
macd, macd_signal, macd_hist = pta.calculate_macd(candles_df['close'])

# Calculate Bollinger Bands
bb_mid, bb_upper, bb_lower =  pta.calculate_bollinger_bands(candles_df['close'])

# Calculate ATR
atr = pta.calculate_atr(candles_df['high'], candles_df['low'], candles_df['close'])

# Calculate ADX
# map_to_one=True is mapped from 0 to 1. map_to_one=False is mapped from 0% to 100%.
adx, adx_plus_di, adx_minus_di = pta.calculate_adx(candles_df['high'], candles_df['low'], candles_df['close'], window=14, map_to_one=True)
```
