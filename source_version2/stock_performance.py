import pandas as pd
import numpy as np
from source_version2.stock_performance import *
from source_version2.stock_features import *
from source_version2.stock_risk import *
from source_version2.stock_strategy import * 
from backtesting import Backtest, Strategy

def calculate_trade_metrics(trade_log, total_bars=252):
    """
    Tính toán 11 chỉ số hiệu suất cấp độ Lệnh (Trade-Level) đầy đủ.
    
    Tham số:
    - trade_log: List các dict kết quả lệnh [{'pnl':.., 'return':.., 'bars':..}, ...]
    - total_bars: Tổng số nến (ngày) của giai đoạn backtest (để tính Time in Market).
    """
    
    df = pd.DataFrame(trade_log)
    
    if df.empty:
        return {"Status": "Không có giao dịch nào"}
        
    # Chuẩn bị 
    winning_trades = df[df['pnl'] > 0]
    losing_trades = df[df['pnl'] <= 0]
    
    n_win = len(winning_trades)
    n_loss = len(losing_trades)
    n_total = n_win + n_loss
    
    # Tính toán các metric 
    gross_gain = winning_trades['pnl'].sum()
    gross_loss = losing_trades['pnl'].abs().sum()
    net_profit = gross_gain - gross_loss
    total_days_in_market = df['bars'].sum()
    time_in_market = total_days_in_market / total_bars
    number_of_trades = n_total
    years = total_bars / 252
    annual_turnover = n_total / years if years > 0 else n_total
    hit_rate = n_win / n_total if n_total > 0 else 0
    profit_factor = gross_gain / gross_loss if gross_loss != 0 else np.inf
    avg_gain = winning_trades['return'].mean() if n_win > 0 else 0
    avg_loss = abs(losing_trades['return'].mean()) if n_loss > 0 else 0
    slugging_ratio = avg_gain / avg_loss if avg_loss != 0 else np.inf

    # --- 3. TRẢ VỀ KẾT QUẢ ---
    metrics = {
        "1. Gross Gain": gross_gain,
        "2. Gross Loss": gross_loss,
        "3. Net Profit": net_profit,
        "4. % Time in Market": time_in_market,
        "5. Number of Trades": number_of_trades,
        "6. Annual Turnover": round(annual_turnover, 2),
        "7. Hit Rate": f"{hit_rate:.2%}",
        "8. Profit Factor": round(profit_factor, 2),
        "9. Avg Gain (Wins)": f"{avg_gain:.2%}",
        "10. Avg Loss (Losses)": f"{avg_loss:.2%}",
        "11. Slugging Ratio": round(slugging_ratio, 2)
    }
    
    return metrics



def get_trade_performance_metrics(ticker_res, total_test_bars):
    
    df_trades = ticker_res['trades']
    
    current_ticker_trades = []
    for _, row in df_trades.iterrows():
        current_ticker_trades.append({
            'pnl': row['PnL'],
            'return': row['ReturnPct'],
            'bars': row['ExitBar'] - row['EntryBar']
        })
    
    metrics = calculate_trade_metrics(current_ticker_trades, total_bars=total_test_bars)
    
    return metrics


def full_get_trade_performance_metrics(all_ticker_results, total_test_bars):
    detailed_rows = []
    
    for res in all_ticker_results:
        m = get_trade_performance_metrics(res, total_test_bars)
        
        # 3. Gắn Ticker vào (nó sẽ nằm ở cuối dictionary)
        m['Ticker'] = res.get('Ticker', 'N/A')
        detailed_rows.append(m)
    df = pd.DataFrame(detailed_rows)
    
    if '3. Net Profit' in df.columns:
        df = df.sort_values(by="3. Net Profit", ascending=False)
    cols = ['Ticker'] + [c for c in df.columns if c != 'Ticker']
    
    return df[cols].reset_index(drop=True)




def calculate_full_portfolio_performance(all_results):
    """
    Tính toán tất cả các chỉ số và trả về Dictionary để dễ dàng xử lý tiếp.
    """
    # 1. Trích xuất Daily Returns
    all_daily_returns = []
    for res in all_results:
        if 'equity_curve' in res:
            daily_ret = res['equity_curve']['Equity'].pct_change().fillna(0)
            all_daily_returns.append(daily_ret)
    
    if not all_daily_returns:
        return {}

    # Gộp thành chuỗi lợi nhuận danh mục
    portfolio_ret = pd.concat(all_daily_returns, axis=1).mean(axis=1)
    r = portfolio_ret.values
    T = len(r)
    
    # 2. Tính toán các chỉ số
    avg_daily_return = np.mean(r)
    annual_avg_return = avg_daily_return * 252
    
    cum_prod = np.prod(1 + r)
    annual_geo_return = (cum_prod ** (252 / T)) - 1 if cum_prod > 0 else -1
    
    daily_vol = np.std(r)
    annual_vol = daily_vol * np.sqrt(252)
    
    # Max Drawdown
    cum_equity = (1 + portfolio_ret).cumprod()
    peak = cum_equity.cummax()
    max_dd = ((cum_equity - peak) / peak).min()

    # 3. Thống kê lệnh
    up_days = r[r > 0]
    down_days = r[r < 0]
    hit_rate = len(up_days) / T
    gross_profit = np.sum(up_days)
    gross_loss = np.sum(np.abs(down_days))
    
    metrics = {
        "total_test_bars": T,
        "avg_daily_return": avg_daily_return,
        "annual_avg_return": annual_avg_return,
        "annual_geo_return": annual_geo_return,
        "daily_vol": daily_vol,
        "annual_vol": annual_vol,
        "max_drawdown": max_dd,
        "sharpe_ratio": annual_avg_return / annual_vol if annual_vol != 0 else 0,
        "calmar_ratio": annual_avg_return / abs(max_dd) if max_dd != 0 else 0,
        "hit_rate": hit_rate,
        "profit_factor": gross_profit / gross_loss if gross_loss != 0 else 0,
        "long_cum_return": cum_prod - 1
    }
    
    return metrics
