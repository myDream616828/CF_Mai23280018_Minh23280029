import pandas as pd
import numpy as np
import os 
from source_version2.stock_features import *
from source_version2.stock_risk import *
from source_version2.stock_strategy import * 
from backtesting import Backtest, Strategy
from source_version2.stock_preparation import * 

#-----------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------
class Bollinger_Mean_Reversion(Strategy):
    # Khai báo tham số để Optimize (Thư viện sẽ dùng các biến này)
    n_param = 20
    k_param = 2
    risk_pct = 0.02
    k_vol_sl = 2
    k_atr = 3
    k_fixed = 0.01
    sl_method = 'atr'

    def init(self):
        df_raw = self.data.df 
       
        def get_indicators(df):
            d = generate_bollinger_band_indicator(df, N=self.n_param, K=self.k_param)
            d = generate_risk_metrics(d)
            d = generate_bollinger_signal(d) 
            return d

        processed = get_indicators(df_raw)
        
        self.sig = self.I(lambda: processed['Signal'], name='Signal')
        self.mid = self.I(lambda: processed['Middle'], name='Middle')
        self.upper = self.I(lambda: processed['Upper'], name='Upper')
        self.lower = self.I(lambda: processed['Lower'], name='Lower')
        self.v_daily = self.I(lambda: processed['Vol_Daily'], name='Vol_Daily')
        self.v_annual = self.I(lambda: processed['Vol_Annual'], name='Vol_Annual')
        self.atr_val = self.I(lambda: processed['ATR'], name='ATR')
 

    def next(self):
        price = self.data.Close[-1]
        signal = self.sig[-1]

        # --- LOGIC VÀO LỆNH (Tối nay có signal, sáng mai khớp) ---
        if not self.position:
            # LONG
            if signal == 1:
                sl = calculate_initial_stop_loss(price, self.sl_method, self.v_daily[-1], self.atr_val[-1], self.k_vol_sl, self.k_atr, self.k_fixed, 1)
                size = calculate_position_size(self.equity, self.risk_pct, self.v_annual[-1], price)
                self.buy(size=size, sl=sl)

            # SHORT
            elif signal == -1:
                sl = calculate_initial_stop_loss(price, self.sl_method, self.v_daily[-1], self.atr_val[-1], self.k_vol_sl, self.k_atr, self.k_fixed, -1)
                size = calculate_position_size(self.equity, self.risk_pct, self.v_annual[-1], price)
                self.sell(size=size, sl=sl)

        # --- LOGIC THOÁT LỆNH (Mean Reversion) ---
        elif self.position.is_long and signal == 0:
            self.position.close() # Mua trả hàng / Bán chốt lời vào sáng mai
            
        elif self.position.is_short and signal == 0:
            self.position.close()

#--------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------
def run_walk_forward_backtest(df_input, train_window=252*5, test_size=252):

    all_stats = []

    for train_data, df_input in split_walk_forward_rolling(df_input, train_window, test_size):
        
        print(f"--- Đang xử lý giai đoạn: {df_input.index[0].date()} đến {df_input.index[-1].date()} ---")
        
        bt_train = Backtest(train_data, Bollinger_Mean_Reversion, cash=1_000_000_000, commission=0.0015, trade_on_close=True)
        

        train_results = bt_train.optimize(
            n_param= [5,10,20,22,50],
            k_param= 2,
            maximize='Return [%]'
        )
        
        best_n = train_results._strategy.n_param
        best_k = train_results._strategy.k_param
        
        class BestParamsStrategy(Bollinger_Mean_Reversion):
            n_param = best_n
            k_param = best_k

        bt_test = Backtest(df_input, BestParamsStrategy, cash=1_000_000_000, commission=0.0015, trade_on_close=True)
        test_stats = bt_test.run()
        
        all_stats.append({
            'period_start': df_input.index[0],
            'period_end': df_input.index[-1],
            'best_n': best_n,
            'best_k': best_k,
            'return': test_stats['Return [%]'],
            'win_rate': test_stats['Win Rate [%]'],
            'trades': test_stats['_trades'] 
        })

    return all_stats
#-----------------------------------------------------------------------------------
""""
def run_simple_backtest(df_input, train_ratio = 0.7):
    all_stats = []
    train_data, df_input = split_simple_train_test(df_input,train_ratio= train_ratio)

    bt_train = Backtest(train_data, Bollinger_Mean_Reversion, cash=1_000_000_000, commission=0.0015, trade_on_close=True)
    train_results = bt_train.optimize(
        n_param = 20,
        k_param = 2,
        maximize = 'Return [%]'
    )

    best_n = train_results._strategy.n_param
    best_k = train_results._strategy.k_param

    class BestParamsStrategy(Bollinger_Mean_Reversion):
            n_param = best_n
            k_param = best_k

    bt_test = Backtest(df_input, BestParamsStrategy, cash=1_000_000_000, commission=0.0015, trade_on_close=True)
    test_stats = bt_test.run()

    all_stats.append({
            'period_start': df_input.index[0],
            'period_end': df_input.index[-1],
            'best_n': best_n,
            'best_k': best_k,
            'return': test_stats['Return [%]'],
            'win_rate': test_stats['Win Rate [%]'],
            'trades': test_stats['_trades'] ,
            #'equity_curve': test_stats['_equity_curve']
        })
    return all_stats

 

#------------------------------------------------------------------
def full_run_simple_backtest(df_all_stocks, train_ratio=0.7):
    
    tickers = df_all_stocks.index.get_level_values(1).unique()
    all_results = []

    print(f"Bắt đầu quét danh mục {len(tickers)} mã...")

    for i, ticker in enumerate(tickers):
        try:
            # 2. Trích xuất dữ liệu của từng mã
            df_ticker = df_all_stocks.xs(ticker, level=1).sort_index()
            
            # 3. Chạy backtest cho mã đó
            ticker_results = run_simple_backtest(df_ticker, train_ratio=train_ratio)
            
            if ticker_results:
                # 4. Giải nén kết quả từ list (ticker_results[0])
                res = ticker_results[0]
                res['Ticker'] = ticker 
                all_results.append(res)
                
                print(f"[{i+1}/{len(tickers)}] ✅ {ticker} hoàn tất.")
                
        except Exception as e:
            print(f" Lỗi tại mã {ticker}: {e}")

    print("\n--- HOÀN TẤT ---")
    return all_results

"""
def run_simple_backtest(df_input,strategy_name="bollinger"):

    StrategyClass = STRATEGY_MAP.get(strategy_name.lower()) 
    if StrategyClass is None:
        raise ValueError(f"Không tìm thấy chiến thuật tên là: {strategy_name}. Các tên hợp lệ: {list(STRATEGY_MAP.keys())}")
    all_stats = []
    bt_data = Backtest(df_input, StrategyClass, cash=1_000_000_000, commission=0.0015, trade_on_close=True)

    test_stats = bt_data.run()

    all_stats.append({
            'period_start': df_input.index[0],
            'period_end': df_input.index[-1],
            'return': test_stats['Return [%]'],
            'win_rate': test_stats['Win Rate [%]'],
            'trades': test_stats['_trades'] ,
            'equity_curve': test_stats['_equity_curve']
        })
    return all_stats 

def full_run_simple_backtest(df_all_stocks,strategy_name="bollinger" ):

    StrategyClass = STRATEGY_MAP.get(strategy_name.lower()) 
    if StrategyClass is None:
        raise ValueError(f"Không tìm thấy chiến thuật tên là: {strategy_name}. Các tên hợp lệ: {list(STRATEGY_MAP.keys())}")
    
    tickers = df_all_stocks.index.get_level_values(1).unique()
    all_results = []

    print(f"Bắt đầu quét danh mục {len(tickers)} mã...")

    for i, ticker in enumerate(tickers):
        try:
            # 2. Trích xuất dữ liệu của từng mã
            df_ticker = df_all_stocks.xs(ticker, level=1).sort_index()
            
            # 3. Chạy backtest cho mã đó
            ticker_results = run_simple_backtest(df_ticker, strategy_name= strategy_name)
            
            if ticker_results:
                # 4. Giải nén kết quả từ list (ticker_results[0])
                res = ticker_results[0]
                res['Ticker'] = ticker 
                all_results.append(res)
                
                print(f"[{i+1}/{len(tickers)}] ✅ {ticker} hoàn tất.")
                
        except Exception as e:
            print(f" Lỗi tại mã {ticker}: {e}")

    print("\n--- HOÀN TẤT ---")
    return all_results

#-------------------------------------------------------------------
def plot_trading_signal(df_input, ticker_name, best_n, best_k, train_ratio=0.7, output_dir = 'result'):
    """
    Hàm vẽ đồ thị kỹ thuật (Candlestick + Bollinger Bands + Signals) cho 1 mã duy nhất.
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Trích xuất dữ liệu của mã đó
        df_ticker = df_input.xs(ticker_name, level=1).sort_index()
        
        # Truyền các tham số 
        class FinalStrategy(Bollinger_Mean_Reversion):
            n_param = best_n
            k_param = best_k

        _, df_input = split_simple_train_test(df_ticker, train_ratio =train_ratio)
        bt_plot = Backtest(df_input, FinalStrategy, 
                           cash=1_000_000_000, 
                           commission=0.0015, 
                           trade_on_close=True)
        
      
        bt_plot.run()
        
        file_name = f"{ticker_name}_N{best_n}_K{best_k}.html"
        file_path = os.path.join(output_dir, file_name)
        
        print(f"Đang vẽ đồ thị mã: {ticker_name}")
        print(f"File HTML sẽ được lưu tại: {file_path}")

        # 7. Lệnh vẽ và lưu file
        # filename: tên file lưu | open_browser: True để tự động mở tab mới
        return bt_plot.plot(filename=file_path, open_browser=True)
        
    except Exception as e:
        print(f"Lỗi khi vẽ mã {ticker_name}: {e}")
#--------------------------------------------------------------
class Turtle_Breakout_Strategy(Strategy):
    
    n_entry = 20
    m_exit = 10
    risk_pct = 0.02
    k_vol_sl = 2
    k_atr = 2    
    k_fixed = 0.01
    sl_method = 'atr'

    def init(self):
        df_raw = self.data.df 
        
        def get_indicators(df):
            d = generate_channel_breakout_indicator(df, n_entry=self.n_entry, m_exit=self.m_exit)
            d = generate_risk_metrics(d,n_atr = 20)
            
            return d
        processed = get_indicators(df_raw)


        self.upper_entry = self.I(lambda: processed['Upper_Entry'], name='Upper_Entry')
        self.lower_entry = self.I(lambda: processed['Lower_Entry'], name='Lower_Entry')
        self.upper_exit = self.I(lambda: processed['Upper_Exit'], name='Upper_Exit')
        self.lower_exit = self.I(lambda: processed['Lower_Exit'], name='Lower_Exit')
        
        self.atr_val = self.I(lambda: processed['ATR'], name='ATR')
        self.v_daily = self.I(lambda: processed['Vol_Daily'], name='Vol_Daily')
        self.v_annual = self.I(lambda: processed['Vol_Annual'], name='Vol_Annual')

    def next(self):

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
   
        if not self.position:
            
            # LONG: Giá High vượt đỉnh Entry (Breakout lên)
            if high > self.upper_entry[-1]:
                # Gọi hàm tính toán Size và SL rời
                sl = calculate_initial_stop_loss(price, self.sl_method, self.v_daily[-1], self.atr_val[-1], self.k_vol_sl, self.k_atr, self.k_fixed, 1)
                size = calculate_position_size(self.equity, self.risk_pct, self.v_annual[-1], price)
                
                self.buy(size=size, sl=sl)

            # SHORT: Giá Low thủng đáy Entry (Breakout xuống)
            elif low < self.lower_entry[-1]:
                sl = calculate_initial_stop_loss(price, self.sl_method, self.v_daily[-1], self.atr_val[-1], self.k_vol_sl, self.k_atr, self.k_fixed, -1)
                size = calculate_position_size(self.equity, self.risk_pct, self.v_annual[-1], price)
                
                self.sell(size=size, sl=sl)

        # --- LOGIC THOÁT LỆNH (EXIT) ---
        # Turtle thoát lệnh khi giá chạm kênh đối diện (Trailing Stop)
        
        elif self.position.is_long:
            # Nếu giá Thấp nhất thủng đáy Exit -> Bán
            if low < self.lower_exit[-1]:
                self.position.close()
                
        elif self.position.is_short:
            # Nếu giá Cao nhất vượt đỉnh Exit -> Mua lại
            if high > self.upper_exit[-1]:
                self.position.close()

STRATEGY_MAP = {
    "bollinger": Bollinger_Mean_Reversion,
    "breakout": Turtle_Breakout_Strategy,
}
