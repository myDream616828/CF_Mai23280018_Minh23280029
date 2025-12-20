import pandas as pd
import pandas_ta as ta
import numpy as np

#1. Cleaning data 
def clean_missing_values(df_in):
    """Xử lý dữ liệu thiếu bằng kỹ thuật Forward Fill."""
    df_clean = df_in.copy()
    df_clean = df_clean.ffill()
    df_clean = df_clean.bfill()
    return df_clean

def clean_outliers_winsorize(df_in, lower=0.01, upper=0.99):
    """Xử lý ngoại lai bằng kỹ thuật Kẹp Lợi Suất (Winsorization)."""
    df_clean = df_in.copy()
    
    # 1. Tính lợi suất
    sr_returns = df_clean['Adj Close'].pct_change()
    
    # 2. Kẹp lợi suất
    limit_low = sr_returns.quantile(lower)
    limit_high = sr_returns.quantile(upper)
    sr_ret_clipped = sr_returns.clip(lower=limit_low, upper=limit_high)
    
    # 3. Tái tạo lại giá
    start_adj = df_clean['Adj Close'].iloc[0]
    start_close = df_clean['Close'].iloc[0]
    
    sr_cum_ret = (1 + sr_ret_clipped).cumprod().fillna(1.0)
    
    df_clean['Adj Close'] = start_adj * sr_cum_ret
    df_clean['Close'] = start_close * sr_cum_ret 
    
    return df_clean

#2. Chia train,test 

def split_walk_forward_rolling(df, train_window, test_size):
    total_rows = len(df)
    
    for i in range(train_window, total_rows, test_size):
        train_end = i
        test_end = min(i + test_size, total_rows)
        
        # Điểm bắt đầu của Train trượt theo i
        train_start = i - train_window
        
        if test_end <= train_end:
            break
            
        train_data = df.iloc[train_start:train_end].copy()
        test_data = df.iloc[train_end:test_end].copy()
        
        yield train_data, test_data

def split_walk_forward_expanding(df, train_size, test_size):

    total_rows = len(df)

    for i in range(train_size, total_rows, test_size):
        train_end = i
        test_end = min(i + test_size, total_rows)
        
        if test_end <= train_end:
            break
            
        train_data = df.iloc[:train_end].copy()
        test_data = df.iloc[train_end:test_end].copy()

        yield train_data, test_data

def split_simple_train_test(df, train_ratio=0.8):
   
    n = len(df)
    split_index = int(n * train_ratio)
    
    train_data = df.iloc[:split_index].copy()
    test_data = df.iloc[split_index:].copy()
    
    return train_data, test_data
