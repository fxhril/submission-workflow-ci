import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import mlflow
import mlflow.sklearn
import dagshub

# --- KONFIGURASI DAGSHUB (Kriteria Advanced) ---
DAGSHUB_USER = os.environ.get("DAGSHUB_USERNAME")
DAGSHUB_REPO = os.environ.get("DAGSHUB_REPO_NAME")
DAGSHUB_TOKEN = os.environ.get("DAGSHUB_TOKEN")

# Setup auth DagsHub
dagshub.auth.add_app_token(DAGSHUB_TOKEN)
dagshub.init(repo_owner=DAGSHUB_USER, repo_name=DAGSHUB_REPO, mlflow=True)

# Ini otomatis setup remote tracking URI
dagshub.init(repo_owner=DAGSHUB_USER, repo_name=DAGSHUB_REPO, mlflow=True)

# Load Data
df = pd.read_csv('heartclean_preprocessing.csv')
X = df.drop(columns=['target']) 
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Enable Autolog (Standar)
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="Hyperparameter_Tuning_Advanced"):
    
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error')
    
    # Train
    grid_search.fit(X_train, y_train)
    
    # Predict
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # Evaluasi
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Best Params: {grid_search.best_params_}")
    print(f"MAE: {mae}")

    # --- TAMBAHAN MANUAL LOGGING (Kriteria Advanced) ---
    
    # Log Metrics tambahan manual
    mlflow.log_metric("manual_mae", mae)
    mlflow.log_metric("manual_mse", mse)
    mlflow.log_metric("manual_r2", r2)
    mlflow.log_metric("training_rows", len(X_train)) # Metrik custom
    
    # Log Params manual (jika perlu eksplisit)
    mlflow.log_param("tuning_method", "GridSearchCV")
    
    # --- ARTEFAK 1: Sample Predictions ---
    comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    comparison_df.head().to_csv("sample_predictions.csv", index=False)
    mlflow.log_artifact("sample_predictions.csv")
    print("Artifact 1 'sample_predictions.csv' berhasil diupload.")

    # --- ARTEFAK 2: Feature Importance (TAMBAHAN BARU) ---
    # Mengambil data fitur mana yang paling berpengaruh terhadap prediksi
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    importance_df.to_csv("feature_importance.csv", index=False)
    mlflow.log_artifact("feature_importance.csv")
    print("Artifact 2 'feature_importance.csv' berhasil diupload.")

    print("Training selesai & pushed to DagsHub.")
