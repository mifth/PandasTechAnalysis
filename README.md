# PandasTechAnalysis
Python tech analysis for markets using Pandas and Numpy

<img width="1512" height="916" alt="image" src="https://github.com/user-attachments/assets/824e51a1-9f96-439c-894a-ba83e14dec32" />

- Get your candles/values data and convert it to Pandas.
- Use PandasTechAnalysis.

Examples:
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

# Stoch RSI
stoch_rsi, stoc_k, stoch_d = pta.calculate_stoch_rsi(candles_df['close'], rsi_14, window=14)

# MACD
macd, macd_signal, macd_hist = pta.calculate_macd(candles_df['close'])

# Calculate Bollinger Bands
bb_mid, bb_upper, bb_lower =  pta.calculate_bollinger_bands(candles_df['close'])

# Calculate ATR
atr = pta.calculate_atr(candles_df['high'], candles_df['low'], candles_df['close'])
```
