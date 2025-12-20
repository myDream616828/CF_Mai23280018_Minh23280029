import pandas as pd
import numpy as np 
# Bollinger_Band_Mean_Reversion 
def generate_bollinger_signal(df):
    df_out = df.copy()

    # Điều kiện vào lệnh
    df_out["Long_entry"]  = (df_out["Close"] < df_out["Lower"]) | ((df_out["Zscore"] < -2) & (df_out["PercentB"] < 0))
    df_out["Short_entry"] = (df_out["Close"] > df_out["Upper"]) | ((df_out["Zscore"] > 2) & (df_out["PercentB"] > 1))

    # Điều kiện thoát lệnh
    df_out["Long_exit"]  = df_out["Close"] > df_out["Middle"]
    df_out["Short_exit"] = df_out["Close"] < df_out["Middle"]

    # Tạo cột signal
    df_out["Signal"] = 0

    position = 0  # 1 = long, -1 = short, 0 = neutral
    signals = []

    for i in range(len(df_out)):
        if position == 0:
            if df_out["Long_entry"].iloc[i]:
                position = 1
            elif df_out["Short_entry"].iloc[i]:
                position = -1

        elif position == 1:
            if df_out["Long_exit"].iloc[i]:
                position = 0

        elif position == -1:
            if df_out["Short_exit"].iloc[i]:
                position = 0

        signals.append(position)

    df_out["Signal"] = signals
    return df_out

def full_generate_bollinger_signal(df):
    df_processed = df.groupby("Ticker", group_keys=False).apply(generate_bollinger_signal)
    return df_processed
