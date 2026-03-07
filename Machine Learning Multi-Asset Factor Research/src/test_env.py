import platform
import numpy as np
import pandas as pd
import sklearn
import shap
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 1. 檢查硬體架構
print("=== Environment Check ===")
print(f"System Architecture: {platform.machine()}")
print(f"Python Version: {platform.python_version()}")
print(f"Pandas Version: {pd.__version__}")
print(f"Scikit-learn Version: {sklearn.__version__}")

# 2. 驗證 M2 運算能力 (簡單矩陣運算)
try:
    # 建立一個大矩陣測試記憶體與運算
    a = np.random.rand(1000, 1000)
    b = np.random.rand(1000, 1000)
    c = np.dot(a, b)
    print("✅ Numpy Matrix Multiplication Test: SUCCESS")
except Exception as e:
    print(f"❌ Numpy Test Failed: {e}")

# 3. 驗證 SHAP 套件載入 (通常這是最容易出錯的環節)
try:
    # 建立一個極簡易的解釋器測試
    X, y = shap.datasets.adult()
    model = sklearn.linear_model.LinearRegression()
    model.fit(X.iloc[:10], y[:10])
    explainer = shap.Explainer(model, X.iloc[:10])
    shap_values = explainer(X.iloc[:2])
    print("✅ SHAP Library Test: SUCCESS")
except Exception as e:
    print(f"❌ SHAP Test Failed: {e}")

print("=== Ready for Development ===")