import pandas as pd
import pandas_ta as ta
import numpy as np
from arch import arch_model

# ========================================================
# 1. NHÓM HÀM LÀM SẠCH (CLEANING FUNCTIONS)
# ========================================================

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

# ========================================================
# 2. HÀM TÍNH TOÁN GARCH (TỐI ƯU HÓA)
# ========================================================

def calc_garch_optimized(df_in, lookback_window=252):
    """
    Dự báo độ biến động GARCH(1,1) với cấu hình tối ưu tốc độ.
    - Mean='Zero': Giả định lợi suất trung bình = 0 (nhanh hơn).
    - Scaling: Nhân 100 để tránh lỗi hội tụ.
    - Rolling Loop: Chạy cuốn chiếu để tránh Look-ahead Bias.
    """
    # Chuẩn bị dữ liệu Log Returns
    # (Dùng shift(1) ở đây là đúng để tính return quá khứ)
    log_returns = np.log(df_in['Adj Close'] / df_in['Adj Close'].shift(1)).dropna()
    
    # Khởi tạo list kết quả (điền NaN cho giai đoạn warm-up)
    sigma_forecasts = [np.nan] * lookback_window
    
    # Scaling dữ liệu (Mẹo quan trọng để GARCH chạy mượt)
    scaling_factor = 100
    scaled_returns = log_returns * scaling_factor
    
    # Chuyển sang numpy array để loop nhanh hơn
    values = scaled_returns.values
    
    print(f"   -> Chạy GARCH Rolling ({len(values)} phiên)...")
    
    # Vòng lặp Rolling Forecast
    # Tại ngày i, ta dùng dữ liệu từ (i-window) đến i để dự báo cho i+1
    for i in range(lookback_window, len(values)):
        window_data = values[i - lookback_window : i]
        
        try:
            # Cấu hình tối ưu: mean='zero', max_iter=75
            am = arch_model(window_data, vol='Garch', p=1, q=1, mean='zero', dist='normal')
            
            # Tắt hiển thị log (disp='off')
            res = am.fit(disp='off', max_iter=75, options={'ftol': 1e-5})
            
            # Dự báo phương sai (Variance) cho ngày tiếp theo (Horizon=1)
            forecast_var = res.forecast(horizon=1).variance.iloc[-1, 0]
            
            # Tính độ lệch chuẩn (Sigma) và scale ngược lại
            sigma = np.sqrt(forecast_var) / scaling_factor
            
        except Exception:
            # Cơ chế Phao cứu sinh: Nếu lỗi, dùng lại giá trị ngày hôm trước
            sigma = sigma_forecasts[-1] if len(sigma_forecasts) > 0 and not np.isnan(sigma_forecasts[-1]) else 0.01
            
        sigma_forecasts.append(sigma)
    
    # Gán lại vào DataFrame (cần căn chỉnh Index với log_returns)
    # sigma_forecasts đang khớp độ dài với log_returns
    sr_garch = pd.Series(sigma_forecasts, index=log_returns.index, name='GARCH_Vol')
    
    # Merge vào df gốc
    df_out = df_in.copy()
    df_out['GARCH_Vol'] = sr_garch
    
    # Fill các giá trị đầu tiên (NaN) bằng backfill
    df_out['GARCH_Vol'] = df_out['GARCH_Vol'].bfill()
    
    return df_out

# ========================================================
# 3. HÀM TÍNH CHỈ BÁO TỔNG HỢP
# ========================================================

def calc_indicators(df_in):
    """Tính toán toàn bộ chỉ báo: Technical + GARCH."""
    df_feat = df_in.copy()
    
    # --- A. Xu hướng & Momentum ---
    df_feat['SMA_50'] = ta.sma(df_feat['Close'], length=50)
    df_feat['SMA_200'] = ta.sma(df_feat['Close'], length=200)
    df_feat['ROC_12M'] = ta.roc(df_feat['Adj Close'], length=252)
    df_feat['ROC_6M'] = ta.roc(df_feat['Adj Close'], length=126)

    # --- B. Đảo chiều ---
    df_feat['RSI_14'] = ta.rsi(df_feat['Close'], length=14)
    
    # Fix lỗi tên cột BBands bằng iloc
    bbands = ta.bbands(df_feat['Close'], length=20, std=2.0)
    if bbands is not None:
        df_feat['BB_Lower'] = bbands.iloc[:, 0] 
        df_feat['BB_Upper'] = bbands.iloc[:, 2]

    # --- C. Xác nhận ---
    df_feat['OBV'] = ta.obv(df_feat['Close'], df_feat['Volume'])
    df_feat['Vol_SMA_20'] = ta.sma(df_feat['Volume'], length=20)

    # --- D. Quản lý rủi ro (ATR Cơ bản) ---
    df_feat['ATR_14'] = ta.atr(df_feat['High'], df_feat['Low'], df_feat['Close'], length=14)
    
    # --- E. Quản lý rủi ro Nâng cao (GARCH) ---
    # Gọi hàm GARCH tối ưu vừa viết
    # (Lưu ý: Hàm này vẫn chậm hơn ATR nhiều, nhưng nhanh hơn GARCH gốc)
    df_feat = calc_garch_optimized(df_feat, lookback_window=252)
    
    return df_feat

# ========================================================
# 4. PIPELINE XỬ LÝ CHÍNH
# ========================================================

def _process_single_ticker(df_ticker):
    """Pipeline cho 1 mã."""
    # 1. Làm sạch
    df_step1 = clean_missing_values(df_ticker)
    df_step2 = clean_outliers_winsorize(df_step1)
    
    # 2. Tính chỉ báo (bao gồm GARCH)
    df_final = calc_indicators(df_step2)
    
    return df_final

def process_all_tickers(df_panel):
    """Hàm gọi từ bên ngoài."""
    print("⚙️ Đang chạy Pipeline: Missing -> Outlier -> Features (GARCH included)...")
    
    # Groupby và Apply
    df_processed = df_panel.groupby('Ticker', group_keys=False).apply(_process_single_ticker)
    
    # Xóa dữ liệu warm-up
    df_processed = df_processed.dropna()
    
    print(f"✅ Hoàn tất! Dữ liệu sạch còn lại: {df_processed.shape[0]} dòng.")
    return df_processed
