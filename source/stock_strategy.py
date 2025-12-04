import pandas as pd
import numpy as np
import statsmodels.api as sm

# ========================================================
# 1. CHIẾN LƯỢC MEAN REVERSION (ĐẢO CHIỀU - TIME SERIES)
# ========================================================
def generate_mean_reversion_signal(df_in):
    """
    Tín hiệu: Mua khi giá thủng Bollinger Band dưới và quay đầu (Inflection).
    """
    df = df_in.copy()
    
    # A. Trend Filter (SMA 200)
    # Chỉ mua khi xu hướng dài hạn là Tăng
    cond_trend_up = df['Close'] > df['SMA_200']
    
    # B. Entry Trigger (Inflection Point)
    # Hôm nay (T): Giá đã chui vào trong dải (Close > BB_Lower)
    # Hôm qua (T-1): Giá nằm ngoài dải (Close < BB_Lower)
    # 🔔 Shift(1) đảm bảo không nhìn trộm tương lai
    cond_inflection = (
        (df['Close'] > df['BB_Lower']) & 
        (df['Close'].shift(1) < df['BB_Lower'].shift(1))
    )
    
    # C. Exit Signal (Chạm SMA 50)
    # Chốt lời khi giá hồi phục về mức trung bình
    cond_exit = df['Close'] >= df['SMA_50']
    
    # Gán tín hiệu
    df['Signal_MR'] = 0
    df.loc[cond_trend_up & cond_inflection, 'Signal_MR'] = 1
    df.loc[cond_exit, 'Signal_MR'] = -1
    
    return df

# ========================================================
# 2. CHIẾN LƯỢC MOMENTUM (ĐÀ TĂNG TRƯỞNG - CROSS SECTIONAL)
# ========================================================
def generate_momentum_signal(df_panel, top_n=7):
    """
    Tín hiệu: Mua Top N mã mạnh nhất (ROC 12M) mỗi tháng.
    """
    # 1. Chuẩn bị ma trận (Ticker thành Cột)
    df_mom = df_panel['ROC_12M'].unstack()
    
    # 2. Resample cuối tháng (Chỉ xếp hạng 1 lần/tháng)
    df_monthly = df_mom.resample('M').last()
    
    # 3. Xếp hạng (Rank)
    # axis=1: So sánh ngang giữa các mã
    df_ranks = df_monthly.rank(axis=1, ascending=False)
    
    # 4. Chọn Top N (Tín hiệu tháng)
    df_monthly_sig = (df_ranks <= top_n).astype(int)
    
    # 5. Kéo giãn ra ngày (Daily)
    # ffill: Giữ nguyên danh mục trong suốt tháng đó
    df_daily_sig = df_monthly_sig.reindex(df_mom.index).ffill()
    
    # 6. Trả về Series (để ghép vào bảng chính)
    sr_signal = df_daily_sig.stack()
    sr_signal.name = 'Signal_MOM'
    
    return sr_signal

# ========================================================
# 3. CHIẾN LƯỢC PAIRS TRADING (GIAO DỊCH CẶP - NÂNG CAO)
# ========================================================
def calculate_spread_zscore(price_A, price_B, window=30):
    """
    Hàm phụ trợ: Tính Z-Score của Spread giữa 2 mã.
    """
    # Dùng Log Price để tính toán chuẩn xác hơn
    log_A = np.log(price_A)
    log_B = np.log(price_B)

    # Tính Hedge Ratio (Beta) bằng OLS tĩnh
    X = sm.add_constant(log_B)
    try:
        model = sm.OLS(log_A, X, missing='drop').fit()
        beta = model.params.iloc[1]
    except:
        return None, None

    # Tính Spread
    spread = log_A - beta * log_B

    # Tính Z-Score (Rolling Window để tránh Bias)
    spread_mean = spread.rolling(window=window).mean()
    spread_std = spread.rolling(window=window).std()
    z_score = (spread - spread_mean) / spread_std
    
    return z_score, beta

def generate_pair_signal(df_panel, stock_A, stock_B, window=30, entry_z=2.0, exit_z=0.5):
    """
    Tạo tín hiệu Pairs Trading: 1 (Long Spread), -1 (Short Spread), 0 (Exit).
    """
    try:
        price_A = df_panel.xs(stock_A, level='Ticker')['Adj Close']
        price_B = df_panel.xs(stock_B, level='Ticker')['Adj Close']
    except KeyError:
        return None, None

    # Tính Z-Score
    z_score, beta = calculate_spread_zscore(price_A, price_B, window)
    if z_score is None: return None, None
    
    # Tạo chuỗi tín hiệu
    signal = pd.Series(0, index=z_score.index, name='Signal_Pair')
    
    # Logic vào lệnh (Mean Reversion trên Spread)
    signal[z_score < -entry_z] = 1   # Z thấp -> Mua A, Bán B (Long Spread)
    signal[z_score > entry_z] = -1   # Z cao -> Bán A, Mua B (Short Spread)
    
    # Logic thoát lệnh (Vùng trung tính)
    exit_mask = abs(z_score) < exit_z
    signal[exit_mask] = 0
    
    # DataFrame kết quả
    df_res = pd.DataFrame({'Z_Score': z_score, 'Signal_Pair': signal})
    return df_res, beta

# ========================================================
# 4. CÁC HÀM CHẠY TỔNG HỢP (RUNNERS)
# ========================================================

def run_mean_reversion_strategy(df_panel):
    print("🧠 Chạy Mean Reversion (Single Stock)...")
    return df_panel.groupby('Ticker', group_keys=False).apply(generate_mean_reversion_signal)

def run_momentum_strategy(df_panel, top_n=7):
    print(f"🧠 Chạy Momentum (Top {top_n})...")
    sr_mom = generate_momentum_signal(df_panel, top_n)
    df_res = df_panel.copy()
    df_res['Signal_MOM'] = sr_mom.fillna(0)
    return df_res

def run_pair_strategy(df_panel, stock_A, stock_B):
    print(f"🧠 Chạy Pairs Trading: {stock_A} - {stock_B}...")
    return generate_pair_signal(df_panel, stock_A, stock_B)

def run_all_strategies(df_panel):
    """
    Chạy 2 chiến lược chính (MeanRev + Momentum) cho toàn bộ danh mục.
    (Pairs Trading chạy riêng vì cần chọn cặp cụ thể).
    """
    print("🧠 Đang chạy TẤT CẢ chiến lược chính...")
    
    # 1. Mean Reversion
    df_res = df_panel.groupby('Ticker', group_keys=False).apply(generate_mean_reversion_signal)
    
    # 2. Momentum
    sr_mom = generate_momentum_signal(df_panel)
    df_res['Signal_MOM'] = sr_mom.fillna(0)
    
    print("✅ Hoàn tất tạo tín hiệu.")
    return df_res
