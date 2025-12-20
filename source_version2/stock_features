# Bollinger_Band
import pandas as pd
import pandas_ta as ta
import numpy as np

def generate_bollinger_band_indicator(df_in, N = 20, K = 2):
    df_out = df_in.copy()
    price = df_out['Close']
    votatility = price.rolling(N).std()
    
    df_out["Middle"] = price.rolling(N).mean()
    df_out["Upper"] = df_out["Middle"] + K *  votatility
    df_out["Lower"] = df_out["Middle"] - K * votatility
    df_out["Zscore"] = (price - df_out["Middle"]) / votatility
    df_out["PercentB"] = (price - df_out["Lower"]) / (df_out["Upper"] - df_out["Lower"])
    return df_out 

def full_generate_bollinger_band_indicator(df):
    df_processed = df.groupby('Ticker', group_keys = False).apply(generate_bollinger_band_indicator)
    return df_processed



def generate_channel_breakout_indicator(df, n_entry=20, m_exit=10):
    df = df.copy()
    df['Upper_Entry'] = df['High'].rolling(n_entry).max().shift(1)
    df['Lower_Entry'] = df['Low'].rolling(n_entry).min().shift(1)
    
    df['Upper_Exit'] = df['High'].rolling(m_exit).max().shift(1)
    df['Lower_Exit'] = df['Low'].rolling(m_exit).min().shift(1)
    return df

def full_generate_channel_breakout_indicator(df):
    df_processed = df.groupby('Ticker', group_keys = False).apply(generate_channel_breakout_indicator)
    return df_processed
