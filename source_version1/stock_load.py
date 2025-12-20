import yfinance as yf
import pandas as pd
import os

# ========================================================
# 1. HÀM ĐỌC DANH SÁCH MÃ (TỪ CSV)
# ========================================================
def load_ticker_list(csv_path='data/tickers_70.csv'):
    """
    Đọc danh sách mã chứng khoán từ file CSV.
    Trả về: List các mã (tickers) và DataFrame thông tin chi tiết.
    """
    if not os.path.exists(csv_path):
        print(f"❌ Lỗi: Không tìm thấy file '{csv_path}'. Vui lòng tạo file này trước.")
        return [], None
    
    try:
        df_info = pd.read_csv(csv_path)
        # Loại bỏ khoảng trắng thừa nếu có
        df_info['Ticker'] = df_info['Ticker'].str.strip()
        
        tickers = df_info['Ticker'].tolist()
        print(f"✅ Đã đọc {len(tickers)} mã từ file CSV.")
        return tickers, df_info
        
    except Exception as e:
        print(f"❌ Lỗi đọc file CSV: {e}")
        return [], None

# ========================================================
# 2. HÀM TẢI DỮ LIỆU GIÁ (DOWNLOADER)
# ========================================================
def download_data(tickers, start_date, end_date, filename='data/stock_data_70.csv'):
    """
    Tải dữ liệu OHLCV từ Yahoo Finance dựa trên danh sách tickers.
    Luôn tải mới và ghi đè file cũ để đảm bảo dữ liệu mới nhất.
    """
    if not tickers:
        print("⚠️ Danh sách mã trống! Không thể tải dữ liệu.")
        return None

    print("------- Quá trình tải dữ liệu (Fresh Download) ----------------")
    print(f"⬇️ Đang tải dữ liệu từ {start_date} đến {end_date}...")
    
    try:
        # auto_adjust=False -> QUAN TRỌNG để giữ Close gốc và Adj Close riêng
        raw_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, group_by='ticker')
        
        if raw_data.empty:
            print("❌ Lỗi: Không tải được dữ liệu nào!")
            return None

        # Xử lý định dạng (Stacking)
        # Chuyển từ Wide Format sang Long Format (MultiIndex: Date, Ticker)
        df_stacked = raw_data.stack(level=0)
        
        # Đặt tên Index
        df_stacked.index.names = ['Date', 'Ticker']
        
        # Chuẩn hóa tên cột (open -> Open)
        df_stacked.columns = [col.title() for col in df_stacked.columns]
        
        # Sắp xếp
        df_stacked = df_stacked.sort_index()

        # Lưu xuống ổ cứng (Tạo thư mục data nếu chưa có)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df_stacked.to_csv(filename)
        
        print(f"💾 Đã lưu dữ liệu mới vào '{filename}'.")
        print("-------------------- Hoàn thành -----------------")
        
        return df_stacked

    except Exception as e:
        print(f"❌ Lỗi nghiêm trọng khi tải: {e}")
        return None

# ========================================================
# 3. HÀM ĐỌC DỮ LIỆU ĐÃ TẢI (LOCAL READER)
# ========================================================
def load_local_data(filename='data/stock_data_70.csv'):
    """
    Chỉ đọc dữ liệu từ file csv đã tải sẵn. Dùng khi chạy Backtest nhiều lần.
    """
    if not os.path.exists(filename):
        print(f"⚠️ Chưa có file dữ liệu '{filename}'. Hãy chạy download_data trước.")
        return None
        
    print(f"📂 Đang đọc dữ liệu từ đĩa: {filename}...")
    df = pd.read_csv(filename, index_col=['Date', 'Ticker'], parse_dates=['Date'])
    return df
