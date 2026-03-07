import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# 1. 載入數據 (沿用之前切好的測試集以示公平)
print("Loading Data...")
# 為了公平比較，我們重新讀取完整特徵集並重新切分，確保資料一致
df = pd.read_parquet("hw03/data/dataset_features.parquet")
df.dropna(inplace=True)

# 準備 X, y
exclude_cols = ['Date', 'ticker', 'Fwd_Ret_5d', 'Target']
df_encoded = pd.get_dummies(df, columns=['sector'], drop_first=True)
feature_cols = [c for c in df_encoded.columns if c not in exclude_cols]

X = df_encoded[feature_cols]
y = df_encoded['Target']

# 切分訓練/測試 (前 80% / 後 20%)
split_point = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

print(f"Model Battle: Random Forest vs XGBoost")
print(f"Training Data: {len(X_train)} | Testing Data: {len(X_test)}")

# 2. Random Forest (複習)
print("\nRetraining Random Forest (Baseline)...")
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"   RF Accuracy: {rf_acc:.4f}")

# 3. XGBoost (Challenger)
print("\nTraining XGBoost (Upgrade)...")
# XGBoost 參數說明：
# learning_rate: 學習率，越低越穩但越慢
# n_estimators: 樹的數量
# max_depth: 樹的深度，通常比 RF 淺 (3-6)
# subsample: 隨機抽樣比例，防止過擬合
xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)
print(f"   XGB Accuracy: {xgb_acc:.4f}")

# 4. 進階指標比較 (AUC)
# Accuracy 只看對錯，AUC 看預測概率的排序能力 (對交易更重要)
rf_prob = rf_model.predict_proba(X_test)[:, 1]
xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

rf_auc = roc_auc_score(y_test, rf_prob)
xgb_auc = roc_auc_score(y_test, xgb_prob)

print("\nFinal Result:")
print(f"Random Forest AUC: {rf_auc:.4f}")
print(f"XGBoost AUC:       {xgb_auc:.4f}")

if xgb_auc > rf_auc:
    print("XGBoost 勝出！模型升級有效。")
    # 儲存最強模型
    joblib.dump(xgb_model, "hw03/models/best_model_xgb.pkl")
else:
    print("Random Forest 依然強勁。")

# 5. 畫圖比較特徵重要性 (XGBoost 版本)
# XGBoost 自帶 plot_importance，這跟 SHAP 不一樣，是看 Split 次數
from xgboost import plot_importance
plt.figure(figsize=(10, 8))
plot_importance(xgb_model, max_num_features=15, height=0.5, title="XGBoost Feature Importance (Weight)")
plt.tight_layout()
plt.savefig("xgb_importance.png")
print("Saved XGBoost importance plot to 'xgb_importance.png'")