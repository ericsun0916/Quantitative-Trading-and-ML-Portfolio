import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import os

# 設定繪圖風格
plt.style.use('ggplot')
# 解決 Mac M2 上 matplotlib 中文或負號顯示問題
plt.rcParams['axes.unicode_minus'] = False 

# 1. 載入模型與數據
print("Loading Model and Test Data...")
if not os.path.exists("hw03/models/rf_model.pkl"):
    print("找不到模型檔案，請先執行 Step 3")
    exit()

rf_model = joblib.load("hw03/models/rf_model.pkl")
X_test = pd.read_parquet("hw03/data/X_test.parquet")

print(f"測試集形狀: {X_test.shape}")

# 為了加快 SHAP 計算，取隨機 500 筆樣本
# 如果您的電腦跑得快，可以嘗試增加到 1000
SAMPLE_SIZE = 500
X_sample = X_test.sample(n=min(SAMPLE_SIZE, len(X_test)), random_state=42)

# 2. 建立解釋器並計算 SHAP 值
print("Calculating SHAP values (Uses M2 CPU)...")
# check_additivity=False 可以避免某些微小的數學誤差導致報錯
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_sample, check_additivity=False)

# 3. 維度修正 (關鍵修復部分)
# Random Forest 分類器通常會回傳兩個 class 的 SHAP 值 (跌, 漲)
# 我們需要提取 Class 1 (預測上漲) 的部分

if isinstance(shap_values, list):
    # 情況 A: 回傳 List [Array_Class0, Array_Class1]
    print("ℹDetected SHAP output format: LIST")
    shap_vals_target = shap_values[1]
elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
    # 情況 B: 回傳 3D Array (Samples, Features, Classes)
    print("ℹDetected SHAP output format: 3D ARRAY")
    shap_vals_target = shap_values[:, :, 1]
else:
    # 情況 C: 其他 (通常是回歸問題或 binary 簡化版)
    print("ℹDetected SHAP output format: 2D ARRAY")
    shap_vals_target = shap_values

# 最終檢查
print(f"SHAP Matrix Shape: {shap_vals_target.shape}")
print(f"Feature Matrix Shape: {X_sample.shape}")

if shap_vals_target.shape != X_sample.shape:
    print("警告：維度依然不符，嘗試轉置矩陣...")
    # 極少數情況需要轉置，但在這裡通常不需要，除非 shap 版本極舊

# 4. 繪製並儲存圖表

# A. Summary Plot (Beeswarm)
plt.figure(figsize=(10, 6))
print("Generating Summary Plot...")
# 這裡我們不使用 show=False，直接讓它畫在當前 figure 上
shap.summary_plot(shap_vals_target, X_sample, show=False)
plt.title("SHAP Summary Plot (Predicting Price UP)", fontsize=14)
plt.tight_layout()
plt.savefig("shap_summary_plot.png", dpi=300, bbox_inches='tight')
plt.close()

# B. Bar Plot (重要性排名)
plt.figure(figsize=(10, 6))
print("Generating Bar Plot...")
shap.summary_plot(shap_vals_target, X_sample, plot_type="bar", show=False)
plt.title("Feature Importance Ranking", fontsize=14)
plt.tight_layout()
plt.savefig("shap_importance_bar.png", dpi=300, bbox_inches='tight')
plt.close()

# C. Dependence Plot (針對自定義因子)
# 確保欄位存在才畫
custom_factors = ["Custom_VARM", "Custom_PVD"]
for factor in custom_factors:
    if factor in X_sample.columns:
        print(f"Generating Dependence Plot for {factor}...")
        plt.figure(figsize=(8, 5))
        shap.dependence_plot(factor, shap_vals_target, X_sample, show=False)
        plt.title(f"Dependence Plot: {factor}", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"shap_dependence_{factor.lower()}.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print(f"找不到特徵 {factor}，跳過繪圖")

print("\nSHAP 分析全部完成！圖片已儲存。")