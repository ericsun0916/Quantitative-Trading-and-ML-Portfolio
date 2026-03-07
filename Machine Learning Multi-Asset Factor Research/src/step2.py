import pandas as pd
import numpy as np
import ta  # Technical Analysis library
import os

# 1. 讀取數據
INPUT_FILE = "hw03/data/market_data_raw.parquet"
OUTPUT_FILE = "hw03/data/dataset_features.parquet"

if not os.path.exists(INPUT_FILE):
    print(f"❌ 找不到 {INPUT_FILE}，請先執行 Step 1")
    exit()

df = pd.read_parquet(INPUT_FILE)
print(f"📂 載入數據: {len(df)} 筆")

# 2. 準備特徵容器
# 為了效能，我們針對每個 Ticker 進行分組運算
# 使用 ta 套件可以快速計算標準指標
feature_dfs = []

# 取得 SPY 作為大盤基準 (用於計算 Beta 和相對強弱)
spy_df = df[df['ticker'] == 'SPY'].set_index('Date')['Close'].sort_index()
# 計算 SPY 的日報酬與波動 (用於自定義因子)
spy_ret = spy_df.pct_change()
spy_vol = spy_df.pct_change().rolling(20).std()

print("⚙️ 開始計算 12+ 特徵與自定義因子...")

for ticker, group in df.groupby('ticker'):
    # 複製一份以免影響原始資料
    g = group.copy().sort_values('Date').set_index('Date')
    
    # 確保資料量足夠計算指標 (至少要能算 200MA)
    if len(g) < 200:
        continue
        
    try:
        # A. 基礎技術指標 (8 個)
        
        # 1. RSI (14)
        g['RSI'] = ta.momentum.RSIIndicator(g['Close'], window=14).rsi()
        
        # 2. MACD
        macd = ta.trend.MACD(g['Close'])
        g['MACD_Diff'] = macd.macd_diff() # 柱狀圖
        
        # 3. Bollinger Bands Width (波動率指標)
        bb = ta.volatility.BollingerBands(g['Close'], window=20, window_dev=2)
        g['BB_Width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
        
        # 4. ATR (風險指標)
        g['ATR'] = ta.volatility.AverageTrueRange(g['High'], g['Low'], g['Close'], window=14).average_true_range()
        # 正規化 ATR (除以股價)，否則高價股 ATR 會超大
        g['ATR_Normalized'] = g['ATR'] / g['Close']
        
        # 5. OBV (成交量指標)
        g['OBV'] = ta.volume.OnBalanceVolumeIndicator(g['Close'], g['Volume']).on_balance_volume()
        # OBV 是累加值，數值太大，改取 20日斜率或變化率
        g['OBV_Slope'] = g['OBV'].diff(20)
        
        # 6. VWAP Distance (乖離率)
        # ta 套件的 VWAP 需要 high, low, close, volume
        vwap = ta.volume.VolumeWeightedAveragePrice(g['High'], g['Low'], g['Close'], g['Volume'], window=20)
        g['VWAP_Dist'] = (g['Close'] - vwap.volume_weighted_average_price()) / vwap.volume_weighted_average_price()
        
        # 7. SMA 偏離度 (取代單純的 Price，讓數值標準化)
        g['SMA_50_Dist'] = (g['Close'] / g['Close'].rolling(50).mean()) - 1
        g['SMA_200_Dist'] = (g['Close'] / g['Close'].rolling(200).mean()) - 1
        
        # 8. ROC (動能)
        g['ROC_20'] = g['Close'].pct_change(20)
        
        # B. 跨市場因子 (Cross-Market)
        
        # 9. Beta (對 SPY 的敏感度)
        # 對齊 index
        asset_ret = g['Close'].pct_change()
        # 只取兩者都有的日期
        aligned_spy_ret = spy_ret.reindex(asset_ret.index)
        
        # Rolling Covariance / Rolling Variance
        rolling_cov = asset_ret.rolling(60).cov(aligned_spy_ret)
        rolling_var = aligned_spy_ret.rolling(60).var()
        g['Beta_60'] = rolling_cov / rolling_var
        
        # 10. Relative Strength (相對於 SPY)
        g['RS_SPY'] = g['Close'] / spy_df.reindex(g.index)
        
        # C. 自定義因子 (Custom Factors) - 作業核心
        
        # 11. Custom Factor 1: 波動率調整後的相對動能 (VARM)
        # (Asset_Ret - SPY_Ret) / Asset_Vol
        # 這能找出「漲得比大盤多，且波動不大」的優質資產
        asset_vol = asset_ret.rolling(20).std()
        g['Custom_VARM'] = (g['ROC_20'] - aligned_spy_ret.rolling(20).sum()) / asset_vol
        
        # 12. Custom Factor 2: 量價背離分數 (PVD)
        # 計算價格與成交量的相關係數。負值代表背離 (價漲量縮 或 價跌量增)
        # 注意：我們加上負號，讓「背離程度越高，數值越大」
        g['Custom_PVD'] = -1 * g['Close'].rolling(20).corr(g['Volume'])
        
        # D. 生成標籤 (Prediction Target)
        # 預測未來 5 天報酬率是否 > 0
        g['Fwd_Ret_5d'] = g['Close'].shift(-5) / g['Close'] - 1
        g['Target'] = (g['Fwd_Ret_5d'] > 0).astype(int)
        
        # 移除計算過程中產生的 NaN (前 200 天)
        g.dropna(inplace=True)
        
        # 將 Index 恢復為欄位
        g = g.reset_index()
        g['ticker'] = ticker
        
        feature_dfs.append(g)
        
    except Exception as e:
        print(f"⚠️ Error calculating features for {ticker}: {e}")

# 3. 合併與儲存
if feature_dfs:
    final_df = pd.concat(feature_dfs, ignore_index=True)
    final_df.to_parquet(OUTPUT_FILE, index=False)
    print("-" * 30)
    print(f"特徵工程完成！")
    print(f"最終特徵集: {len(final_df)} 筆數據")
    print(f"儲存至: {OUTPUT_FILE}")
    print(f"包含欄位: {list(final_df.columns)}")
else:
    print("所有資產計算失敗，請檢查數據源。")