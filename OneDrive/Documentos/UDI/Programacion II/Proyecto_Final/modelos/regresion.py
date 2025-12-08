import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


def run_regresion(data_path: str) -> dict:
    df = pd.read_csv(data_path, encoding="latin-1")

    # Seleccionar columnas NUMÉRICAS del dataset
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Columna objetivo
    objetivo = "Total Streams (Millions)"
    if objetivo not in df.columns:
        raise ValueError(f"La columna '{objetivo}' no existe en el dataset.")

    # Variables independientes
    X = df[[c for c in numeric_cols if c != objetivo]]
    y = df[objetivo]

    # Separar datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Entrenar modelo
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predicciones
    y_pred = model.predict(X_test)

    # Métricas
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return {
        "tipo": "Regresión Lineal – Spotify",
        "columnas_entrada": list(X.columns),
        "columna_objetivo": objetivo,
        "MAE": round(mae, 3),
        "MSE": round(mse, 3),
        "RMSE": round(rmse, 3),
        "R2": round(r2, 3),
    }