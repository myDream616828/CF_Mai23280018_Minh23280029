import pandas as pd
import numpy as np

def generate_risk_metrics(df, n_atr=14, n_vol=20):
    df_risk = df.copy()
    
    #a. ATR
    prev_close = df_risk['Close'].shift(1)
    tr1 = df_risk['High'] - df_risk['Low']
    tr2 = (df_risk['High'] - prev_close).abs()
    tr3 = (df_risk['Low'] - prev_close).abs()
    
    df_risk['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df_risk['ATR'] = df_risk['TR'].rolling(n_atr).mean()
    
    #b. VOLATILITY 
    df_risk['Log_Return'] = np.log(df_risk['Close'] / df_risk['Close'].shift(1))
    df_risk['Vol_Daily'] = df_risk['Log_Return'].rolling(n_vol).std()
    df_risk['Vol_Annual'] = df_risk['Vol_Daily'] * np.sqrt(252)
    

    df_risk['ATR_Prev'] = df_risk['ATR'].shift(1)
    df_risk['Vol_Daily_Prev'] = df_risk['Vol_Daily'].shift(1)
    df_risk['Vol_Annual_Prev'] = df_risk['Vol_Annual'].shift(1)
    
    return df_risk
def full_generate_risk_metrics(df, n_atr = 14, n_vol = 20):
    df_processed = df.groupby("Ticker",group_keys=False).apply(generate_risk_metrics, n_atr=n_atr, n_vol=n_vol)
    return df_processed


def calculate_position_size(account_size, risk_pct, vol_annual, price, m=2.0):

    risk_amount = account_size * risk_pct
    vol_impact = (vol_annual * m * price) + 1e-9
    
    # Tính khối lượng
    size = risk_amount / vol_impact

    return int(size)


def calculate_initial_stop_loss(entry_price, method ='sigma', vol_daily= None, atr = None, k_vol = 2, k_atr = 3, k_fixed = 0.01, position_type=1):
    stop_loss_price = 0
    distance = 0
    
    if method == 'sigma':
        distance = k_vol * vol_daily

    elif method == 'atr':
        distance = k_atr * atr

    elif method == 'fixed_pct':
        distance = entry_price * k_fixed
  
    if position_type == 1:  # LONG
        stop_loss_price = entry_price - distance
        
    elif position_type == -1: # SHORT
        stop_loss_price = entry_price + distance
        
    return stop_loss_price
