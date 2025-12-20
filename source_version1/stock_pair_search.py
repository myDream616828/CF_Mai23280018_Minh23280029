import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import stock_load as loader 

# ========================================================
# 1. HÀM PHÂN CỤM (CLUSTERING - UNSUPERVISED LEARNING)
# ========================================================
def get_stock_clusters(df_panel, n_clusters=11):
    """Phân nhóm cổ phiếu dựa trên biến động lợi suất (K-Means)."""
    print(f"⚙️ Đang phân cụm 70 mã thành {n_clusters} nhóm (K-Means)...")
    
    df_returns = df_panel['Adj Close'].unstack().pct_change().dropna()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_returns.T)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    clusters = pd.Series(kmeans.labels_, index=df_returns.columns, name='Cluster')
    print("   -> Kết quả phân bố các cụm:")
    print(clusters.value_counts().sort_index())
    
    return clusters.to_frame()

# ========================================================
# 2. HÀM TÌM CẶP TRONG CÁC NHÓM (CORE LOGIC)
# Dùng chung cho cả Cluster và Sector 
# ========================================================
def find_pairs_in_groups(df_panel, group_df, group_col_name, p_value_threshold=0.05):
    """
    Hàm tìm cặp tổng quát: Chỉ quét các cặp nằm trong cùng một nhóm.
    Input:
        - group_df: DataFrame chứa thông tin nhóm (Index=Ticker, Col=group_col_name)
    """
    # Lấy giá Adj Close
    df_prices = df_panel['Adj Close'].unstack().dropna(axis=1, thresh=int(len(df_panel)*0.9))
    df_prices = df_prices.fillna(method='ffill').fillna(method='bfill')
    
    pairs_list = []
    unique_groups = group_df[group_col_name].unique()
    
    print(f"\n🔍 Bắt đầu quét Cointegration trong từng nhóm '{group_col_name}'...")
    
    for gid in unique_groups:
        # Lấy danh sách mã trong nhóm
        tickers = group_df[group_df[group_col_name] == gid].index.tolist()
        valid_tickers = [t for t in tickers if t in df_prices.columns]
        n = len(valid_tickers)
        
        if n < 2: continue
            
        # Vét cạn các cặp trong nhóm
        for i in range(n):
            for j in range(i + 1, n):
                stock_A = valid_tickers[i]
                stock_B = valid_tickers[j]
                
                s1 = df_prices[stock_A]
                s2 = df_prices[stock_B]
                
                try:
                    _, pvalue, _ = coint(s1, s2)
                    if pvalue < p_value_threshold:
                        pairs_list.append({
                            'Stock_A': stock_A,
                            'Stock_B': stock_B,
                            'P_Value': pvalue,
                            'Group_Type': group_col_name,
                            'Group_ID': gid
                        })
                except: continue

    df_res = pd.DataFrame(pairs_list)
    if not df_res.empty:
        df_res = df_res.sort_values(by='P_Value', ascending=True).reset_index(drop=True)
    
    return df_res

# ========================================================
# 3. CÁC HÀM CHẠY PIPELINE (RUNNERS)
# ========================================================

def run_search_by_cluster(df_data, n_clusters=11):
    """Pipeline 1: Tìm cặp theo Cụm hành vi (K-Means)."""
    print("\n--- CHIẾN LƯỢC 1: CLUSTERING ---")
    df_clusters = get_stock_clusters(df_data, n_clusters)
    # Gọi hàm chung
    return find_pairs_in_groups(df_data, df_clusters, 'Cluster')

def run_search_by_sector(df_data, csv_path='data\tickers_70.csv'):
    """Pipeline 2: Tìm cặp theo Ngành kinh tế (Sector)."""
    print("\n--- CHIẾN LƯỢC 2: SECTOR ---")
    
    _, df_info = loader.load_ticker_list(csv_path)
    if df_info is None: return pd.DataFrame()
    
    # Chuẩn hóa input cho hàm chung
    df_sectors = df_info.set_index('Ticker')[['Sector']]
    
    # Gọi hàm chung (đã sửa tên hàm đúng)
    return find_pairs_in_groups(df_data, df_sectors, 'Sector')


