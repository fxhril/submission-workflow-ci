import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import os

# --- KONFIGURASI ---
# Nama file model yang akan disimpan
MODEL_NAME = "model_random_forest.pkl"

# 1. Load Data
# Pastikan ada file dataset.csv di folder yang sama
os.path.exists('heartclean_preprocessing.csv'):
df = pd.read_csv('heartclean_preprocessing.csv')
    
X = df.drop(columns=['target'])
y = df['target']

# 2. Training
print("Mulai Training Model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=10, random_state=42)
rf.fit(X_train, y_train)

# Evaluasi
y_pred = rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae}")

# 3. SAVE MODEL KE LOKAL (PENTING UNTUK LFS)
joblib.dump(rf, MODEL_NAME)
print(f"Model berhasil disimpan ke file: {MODEL_NAME}")
