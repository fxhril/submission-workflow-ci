import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow

# 1. Load Data
df = pd.read_csv('heartclean_preprocessing.csv')
X = df.drop('target', axis=1)      
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    mlflow.autolog()

    # Training Model (Tanpa Tuning untuk Basic)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy Basic: {acc}")

print("Selesai. Coba cek lokal mlflow.")
