import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import mlflow
import mlflow.sklearn
import dagshub
import joblib
import shutil
import os

# --- 1. SETUP DAGSHUB ---
DAGSHUB_USER = os.environ.get("DAGSHUB_USERNAME")
DAGSHUB_REPO = os.environ.get("DAGSHUB_REPO_NAME")
DAGSHUB_TOKEN = os.environ.get("DAGSHUB_TOKEN")

try:
    dagshub.auth.add_app_token(DAGSHUB_TOKEN)
    dagshub.init(repo_owner=DAGSHUB_USER, repo_name=DAGSHUB_REPO, mlflow=True)
except Exception as e:
    print(f"Info: Berjalan tanpa DagsHub auth. Error: {e}")

# --- 2. PREPARE DATA ---
if os.path.exists('heartclean_preprocessing.csv'):
    df = pd.read_csv('heartclean_preprocessing.csv')
    X = df.drop(columns=['target'])
    y = df['target']
else:
    # Dummy data agar tidak error saat testing
    print("Dataset tidak ditemukan, menggunakan dummy.")
    df = pd.DataFrame({'f1':[1,2], 'f2':[2,1], 'target':[10,20]})
    X = df[['f1','f2']]
    y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. TRAINING & SAVING ---
print("Memulai Training...")

with mlflow.start_run():
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mlflow.log_metric("mae", mae)
    print(f"MAE: {mae}")

    # A. Save ke DagsHub (Syarat Tracking)
    mlflow.sklearn.log_model(rf, "model")
    
    # B. Save Fisik .pkl (Syarat Skilled - LFS)
    joblib.dump(rf, "model_random_forest.pkl")
    print("Model .pkl berhasil disimpan untuk LFS.")

    # C. Save Folder MLflow Lokal (Syarat Advanced - build-docker)
    # Perintah 'mlflow models build-docker' butuh struktur folder MLflow lokal
    if os.path.exists("model_final"):
        shutil.rmtree("model_final") # Hapus dulu kalau ada biar bersih
    mlflow.sklearn.save_model(rf, "model_final")
    print("Folder model_final berhasil dibuat untuk Docker Build.")
