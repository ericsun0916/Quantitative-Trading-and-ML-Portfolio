import ssl
# 1. SSL Bypass (Mac M1/M2 必備)
ssl._create_default_https_context = ssl._create_unverified_context

import yfinance as yf
import pandas as pd
import numpy as np
import os

# 設定儲存目錄
OUTPUT_DIR = "hw03/data"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 2. 定義資產清單 (30+ 檔)
tickers = {
    'Indices': ['SPY', 'QQQ', 'TLT', 'GLD', '^VIX'],
    'Crypto': ['BTC-USD', 'ETH-USD'],
    'Tech': ['NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AMD', 'AVGO', 'CRM', 'NFLX', 'INTC', 'QCOM'],
    'Defensive': ['JPM', 'BAC', 'XOM', 'CVX', 'JNJ', 'PFE', 'KO', 'MCD', 'WMT', 'DIS']
}

# 展平 ticker list
all_tickers = [t for category in tickers.values() for t in category]

# 3. 下載參數
start_date = '2020-01-01'
end_date = '2025-10-31'

print(f"開始下載 {len(all_tickers)} 檔資產數據...")
print(f"區間: {start_date} ~ {end_date}")

# 4. 批量下載 (Auto-adjust for splits)
# group_by='ticker' 讓結構變成 (Ticker -> OHLCV)
raw_data = yf.download(all_tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)

# 5. 數據清洗與格式化 (Flatten MultiIndex)
processed_frames = []

for t in all_tickers:
    try:
        # 提取單一資產數據
        df = raw_data[t].copy()
        
        # 檢查是否下載成功 (有些可能下市或代碼錯誤)
        if df.empty:
            print(f"Warning: No data for {t}")
            continue
            
        # 重新命名欄位 (小寫, 去空白)
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]
        
        # 確保有 Volume 欄位 (指數如 VIX 可能沒有成交量)
        if 'volume' not in df.columns:
            df['volume'] = 0
            
        # 加入識別欄位
        df['ticker'] = t
        
        # 簡單分類標籤 (供後續分析用)
        for cat, t_list in tickers.items():
            if t in t_list:
                df['sector'] = cat
                break
        
        # 重設索引，將 Date 變成欄位
        df = df.reset_index()
        processed_frames.append(df)
        
    except Exception as e:
        print(f"❌ Error processing {t}: {e}")

# 6. 合併與排序
final_df = pd.concat(processed_frames, ignore_index=True)

# 統一欄位名稱 (Yahoo 有時會給 Adj Close 或 Close)
# auto_adjust=True 後，Close 已經是還原權值，Open/High/Low 也是
final_df.rename(columns={'date': 'Date', 'close': 'Close', 'high': 'High', 'low': 'Low', 'open': 'Open', 'volume': 'Volume'}, inplace=True)

# 排序 (對計算技術指標很重要)
final_df.sort_values(by=['ticker', 'Date'], inplace=True)

# 7. 儲存
output_path = f"{OUTPUT_DIR}/market_data_raw.parquet"
final_df.to_parquet(output_path, index=False)

print("-" * 30)
print(f"下載完成！")
print(f"總筆數: {len(final_df)}")
print(f"檔案已儲存至: {output_path}")
print("前 5 筆數據預覽:")
print(final_df.head())