import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib  # 用於儲存模型
import os

# 設定繪圖風格
plt.style.use('ggplot')

# 1. 載入數據
INPUT_FILE = "hw03/data/dataset_features.parquet"
if not os.path.exists(INPUT_FILE):
    print(f"找不到 {INPUT_FILE}")
    exit()

df = pd.read_parquet(INPUT_FILE)
print(f"載入數據: {len(df)} 筆")

# 2. 數據前處理 (Preprocessing)
# ----------------------------------------------------
# A. 去除空值 (Rolling window 產生的 NaN)
df.dropna(inplace=True)

# B. 定義特徵欄位 (X) 與 目標欄位 (y)
# 我們需要排除非數值欄位 (Date, ticker) 和未來數據 (Fwd_Ret_5d, Target)
exclude_cols = ['Date', 'ticker', 'Fwd_Ret_5d', 'Target']

# C. 處理類別特徵 (Sector) - 使用 One-Hot Encoding
# 這讓模型知道 "Tech" 和 "Crypto" 是不同類別
df_encoded = pd.get_dummies(df, columns=['sector'], drop_first=True)

# 重新定義 X 的欄位名稱 (排除排除欄位)
feature_cols = [c for c in df_encoded.columns if c not in exclude_cols]

X = df_encoded[feature_cols]
y = df_encoded['Target']

print(f"使用特徵數: {len(feature_cols)}")
print(f"特徵列表: {feature_cols[:5]} ...")

# 3. 數據分割 (Time-Series Split)
# ----------------------------------------------------
# 金融數據不能隨機打亂 (No Shuffle)，必須按時間切分
# 前 80% 做訓練，後 20% 做測試
split_point = int(len(X) * 0.8)

X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

print(f"訓練集大小: {len(X_train)} | 測試集大小: {len(X_test)}")

# 4. 模型訓練 (Model Training)
# ----------------------------------------------------

# --- Model A: Decision Tree (Baseline) ---
print("\nTraining Decision Tree...")
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

# --- Model B: Random Forest (Main Model) ---
# n_jobs=-1 會調用 M2 晶片的所有核心進行平行運算
print("Training Random Forest (this may take a moment)...")
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42)
rf_model.fit(X_train, y_train)

# 5. 模型評估 (Evaluation)
# ----------------------------------------------------
def evaluate_model(model, name, X_t, y_t):
    pred = model.predict(X_t)
    acc = accuracy_score(y_t, pred)
    print(f"\n=== {name} Performance (Out-of-Sample) ===")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_t, pred))
    return acc

acc_dt = evaluate_model(dt_model, "Decision Tree", X_test, y_test)
acc_rf = evaluate_model(rf_model, "Random Forest", X_test, y_test)

# 6. 儲存模型與測試數據 (供 Step 4 使用)
# ----------------------------------------------------
output_dir = "hw03/models"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

joblib.dump(rf_model, f"{output_dir}/rf_model.pkl")
# 我們把測試集存起來，下一步 SHAP 分析要用
X_test.to_parquet(f"hw03/data/X_test.parquet") 

print("\n模型訓練完成！")
if acc_rf > 0.5:
    print("Random Forest 預測能力優於隨機猜測 (>50%)")
else:
    print("模型預測能力較弱，可能需要更多特徵或調整參數")