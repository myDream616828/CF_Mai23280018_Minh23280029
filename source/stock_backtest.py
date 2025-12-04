import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy

# ==============================================================================
# CHIẾN LƯỢC 1: MEAN REVERSION (CÓ STOP LOSS ĐỘNG & RISK SIZING)
# ==============================================================================
class MeanReversionStrategy(Strategy):
    """
    Chiến lược Đảo chiều: 
    - Mua khi có Signal=1.
    - Bán khi Signal=-1 HOẶC chạm Stop Loss (ATR).
    - Quản lý vốn: Rủi ro 1% vốn/lệnh.
    """
    # --- CẤU HÌNH BƯỚC 4 ---
    risk_per_trade = 0.01      # Rủi ro 1% tài khoản cho mỗi lệnh thua
    sl_atr_multiplier = 2.0    # Cắt lỗ tại khoảng cách 2 ATR
    
    def init(self):
        # Khai báo indicator để backtesting.py vẽ biểu đồ và truy cập dữ liệu
        self.signal = self.I(lambda x: x, self.data.Signal_MR, name='Signal_MR')
        self.atr = self.I(lambda x: x, self.data.ATR_14, name='ATR')
        
    def next(self):
        # Lấy giá trị nến hiện tại
        price = self.data.Close[-1]
        atr = self.atr[-1]
        signal = self.signal[-1]
        
        # --- XỬ LÝ MUA (ENTRY) ---
        # Mua nếu có Tín hiệu và chưa có vị thế
        if signal == 1 and not self.position:
            
            # 1. Tính Giá Cắt Lỗ (Stop Loss Price)
            sl_distance = atr * self.sl_atr_multiplier
            sl_price = price - sl_distance
            
            # 2. Tính Quy mô Vị thế (Position Sizing) theo Rủi ro
            # Số lượng = (Tổng vốn * 1%) / (Khoảng cách SL)
            if sl_distance > 0:
                risk_money = self.equity * self.risk_per_trade
                size_to_buy = risk_money / sl_distance
                
                # Làm tròn xuống số nguyên
                size_to_buy = int(size_to_buy)
                
                # 3. Đặt lệnh MUA kèm SL
                if size_to_buy > 0:
                    # trade_on_close=False (mặc định) -> Sẽ mua giá Open ngày mai
                    self.buy(size=size_to_buy, sl=sl_price)

        # --- XỬ LÝ BÁN (EXIT) ---
        # Bán nếu có tín hiệu thoát (-1) từ chiến lược (ví dụ chạm SMA 50)
        elif signal == -1 and self.position:
            self.position.close()

# ==============================================================================
# CHIẾN LƯỢC 2: MOMENTUM (REBALANCING - TÁI CÂN BẰNG)
# ==============================================================================
class MomentumStrategy(Strategy):
    """
    Chiến lược Đà tăng trưởng:
    - Mua và Giữ (Hold) khi Signal=1 (Nằm trong Top 7).
    - Bán khi Signal=0 (Rớt hạng).
    - Không dùng Stop Loss động, thoát lệnh dựa trên xếp hạng tháng.
    """
    def init(self):
        self.signal = self.I(lambda x: x, self.data.Signal_MOM, name='Signal_MOM')
        
    def next(self):
        signal = self.signal[-1]
        
        # --- REBALANCING LOGIC ---
        
        # Trường hợp 1: Tín hiệu MUA/GIỮ (1)
        if signal == 1:
            if not self.position:
                # Nếu chưa có hàng thì Mua
                # Momentum thường mua All-in số vốn được cấp cho mã đó (hoặc 95%)
                self.buy(size=0.95) 
            
            # Nếu đang có hàng (position) thì Giữ nguyên (Do nothing)
            
        # Trường hợp 2: Tín hiệu BÁN (0)
        elif signal == 0:
            if self.position:
                # Nếu đang có hàng mà tín hiệu mất -> Bán hết
                self.position.close()

# ==============================================================================
# HÀM CHẠY BACKTEST CHO SINGLE STOCK (RUNNERS)
# ==============================================================================

def run_backtest_mean_reversion(df_signals, initial_capital=100_000):
    print(f"🚀 Backtest Mean Reversion (Risk {MeanReversionStrategy.risk_per_trade*100}%, SL {MeanReversionStrategy.sl_atr_multiplier}ATR)...")
    return _run_backtest_loop(df_signals, MeanReversionStrategy, initial_capital, 'Signal_MR')

def run_backtest_momentum(df_signals, initial_capital=100_000):
    print(f"🚀 Backtest Momentum (Rebalancing Top 7)...")
    return _run_backtest_loop(df_signals, MomentumStrategy, initial_capital, 'Signal_MOM')

def _run_backtest_loop(df_panel, strategy_class, capital, signal_col):
    results_list = []
    
    for ticker, df_ticker in df_panel.groupby('Ticker'):
        try:
            df_bt = df_ticker.reset_index(level='Ticker', drop=True).dropna()
            if df_bt.empty or signal_col not in df_bt.columns: continue

            bt = Backtest(
                df_bt, 
                strategy_class,
                cash=capital,
                commission=0.0015,
                trade_on_close=False
            )
            stats = bt.run()
            
            # --- TÍNH TOÁN TIỀN TƯƠI ---
            final_equity = stats['Equity Final [$]']
            net_profit = final_equity - capital  # Lời/Lỗ ròng
            
            results_list.append({
                'Ticker': ticker,
                'Net Profit ($)': net_profit,       # <--- Cột mới quan trọng nhất
                'Final Equity ($)': final_equity,   # <--- Tổng tài sản cuối cùng
                'Return (%)': stats['Return [%]'],
                'Win Rate (%)': stats['Win Rate [%]'],
                'Max Drawdown (%)': stats['Max. Drawdown [%]'],
                '# Trades': stats['# Trades']
            })
        except Exception: pass

    if not results_list: return pd.DataFrame()
    
    df_results = pd.DataFrame(results_list)
    
    # Sắp xếp theo Số tiền lời (Net Profit) giảm dần
    df_results = df_results.sort_values(by='Net Profit ($)', ascending=False)
    
    return df_results


# ==============================================================================
# HÀM BACKTEST RIÊNG CHO PAIRS TRADING (VECTORIZED)
# ==============================================================================

def run_backtest_pair(df_panel, df_signal, stock_A, stock_B, beta, initial_capital=100_000):
    """
    Backtest chiến lược cặp: Long A + Short B (hoặc ngược lại).
    Sử dụng phương pháp Vector hóa để tính PnL.
    Trả về DataFrame chi tiết từng ngày để vẽ biểu đồ.
    """
    print(f"🚀 Backtest Pair: {stock_A} (Long/Short) & {stock_B} (Hedge)...")
    
    # 1. Lấy dữ liệu giá Adj Close để tính lợi nhuận thật
    try:
        price_A = df_panel.xs(stock_A, level='Ticker')['Adj Close']
        price_B = df_panel.xs(stock_B, level='Ticker')['Adj Close']
    except KeyError:
        print("❌ Không tìm thấy dữ liệu giá.")
        return None
        
    # 2. Tính Lợi suất hàng ngày (Daily Returns)
    ret_A = price_A.pct_change()
    ret_B = price_B.pct_change()
    
    # 3. Chuẩn bị khung dữ liệu chung
    # Inner join để đảm bảo cùng khung thời gian
    df_bt = pd.DataFrame(index=price_A.index)
    df_bt['Ret_A'] = ret_A
    df_bt['Ret_B'] = ret_B
    
    # Ghép tín hiệu vào (Lưu ý: Tín hiệu T dùng cho Return T+1)
    # Shift(1) ở đây là để khớp lệnh vào ngày hôm sau
    df_bt['Signal'] = df_signal['Signal_Pair'].shift(1)
    
    # 4. Tính Lợi nhuận Chiến lược (Strategy Return)
    # Logic: 
    # - Nếu Signal = 1 (Long Spread): Mua A, Bán B (theo tỷ lệ beta)
    # - Nếu Signal = -1 (Short Spread): Bán A, Mua B
    
    # PnL = Signal * (Ret_A - Beta * Ret_B)
    # (Giả định ta phân bổ vốn vào A và hedge bằng B)
    df_bt['Strat_Ret'] = df_bt['Signal'] * (df_bt['Ret_A'] - beta * df_bt['Ret_B'])
    
    # Trừ phí giao dịch (giả định 0.15% mỗi lần vào/ra lệnh)
    # Phí tính mỗi khi Signal thay đổi (đảo vị thế)
    trades = df_bt['Signal'].diff().abs()
    df_bt['Strat_Ret'] = df_bt['Strat_Ret'] - (trades * 0.003) # 0.15% * 2 (Mua + Bán)
    
    # 5. Tính Đường cong vốn (Equity Curve)
    df_bt['Equity_Curve'] = (1 + df_bt['Strat_Ret'].fillna(0)).cumprod() * initial_capital
    
    # 6. Thống kê Kết quả
    final_equity = df_bt['Equity_Curve'].iloc[-1]
    net_profit = final_equity - initial_capital
    total_return = (net_profit / initial_capital) * 100
    
    # Drawdown
    peak = df_bt['Equity_Curve'].cummax()
    drawdown = (df_bt['Equity_Curve'] - peak) / peak
    max_dd = drawdown.min() * 100
    
    print(f"   💰 Lợi nhuận ròng: ${net_profit:,.2f}")
    print(f"   📈 Tỷ suất sinh lời: {total_return:.2f}%")
    print(f"   📉 Max Drawdown: {max_dd:.2f}%")
    
    # Trả về DataFrame đầy đủ để vẽ biểu đồ
    return df_bt
