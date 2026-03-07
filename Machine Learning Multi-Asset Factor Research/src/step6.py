import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# 1. 載入模型與數據
rf_model = joblib.load("hw03/models/rf_model.pkl")
df_full = pd.read_parquet("hw03/data/dataset_features.parquet")

# 切分測試集
split_point = int(len(df_full) * 0.8)
test_df = df_full.iloc[split_point:].copy()

# 準備 X
test_df_encoded = pd.get_dummies(test_df, columns=['sector'], drop_first=True)
model_features = rf_model.feature_names_in_
for col in model_features:
    if col not in test_df_encoded.columns:
        test_df_encoded[col] = 0
X_test = test_df_encoded[model_features]

# 2. 生成機率
probs = rf_model.predict_proba(X_test)
prob_up = probs[:, 1] # 預測上漲的機率

# ==========================================
# 3. 策略優化 (Strategy Optimization)
# ==========================================

# 調整 A: 降低門檻 (從 0.53 -> 0.51)
THRESHOLD = 0.51 

test_df['Signal'] = 0
# 調整 B: 只做多 (Long Only) - 在牛市中測試選股能力
test_df.loc[prob_up > THRESHOLD, 'Signal'] = 1 

# 檢查訊號分佈 (Debug 關鍵)
signal_counts = test_df['Signal'].value_counts()
print("\n訊號分佈狀況:")
print(signal_counts)
if 1 not in signal_counts:
    print("警告：沒有任何 '做多' 訊號！請再降低門檻 (例如 0.505)")

# 4. 計算回報
# 假設持有 5 天，每天獲得 1/5 的報酬 (簡化計算)
# 或是直接看 Fwd_Ret_5d (這是 5 天後的總報酬)
# 為了畫日線圖，我們將 5 天報酬平攤到每天 (Approximation)
daily_ret = test_df['Fwd_Ret_5d'] / 5 

# 扣除交易成本 (假設 0.1%)
COST = 0.001
test_df['Strategy_Ret'] = test_df['Signal'] * daily_ret
# 簡單扣成本：只要有持倉就扣一點摩擦成本 (保守估計)
test_df.loc[test_df['Signal'] == 1, 'Strategy_Ret'] -= (COST / 5) 

# 5. 投資組合績效 (每日平均)
portfolio_returns = test_df.groupby('Date')['Strategy_Ret'].mean()
benchmark_returns = test_df.groupby('Date')['Fwd_Ret_5d'].mean() / 5

# 累積報酬
cum_strategy = (1 + portfolio_returns).cumprod()
cum_benchmark = (1 + benchmark_returns).cumprod()

# 6. 繪圖
plt.figure(figsize=(12, 6))
plt.plot(cum_strategy, label=f'AI Strategy (Long Only, Thresh>{THRESHOLD})', color='blue', linewidth=2)
plt.plot(cum_benchmark, label='Market Average (Benchmark)', color='grey', linestyle='--', alpha=0.6)

plt.title(f"Optimized Backtest: AI vs Market (Threshold {THRESHOLD})")
plt.xlabel("Date")
plt.ylabel("Cumulative Return (1.0 = Start)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("backtest_optimized.png")
print("優化版回測圖已儲存至 'backtest_optimized.png'")
plt.show()