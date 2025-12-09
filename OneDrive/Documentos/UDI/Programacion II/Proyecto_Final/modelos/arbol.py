import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_IMG_DIR = os.path.join(BASE_DIR, "static", "img")
os.makedirs(STATIC_IMG_DIR, exist_ok=True)


def run_arbol(data_path=None):
    try:
        if data_path is None:
            data_path = os.path.join(DATA_DIR, "Spotify_2024_Global_Streaming_Data.csv")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"No se encontró el dataset en: {data_path}")

        df = pd.read_csv(data_path)

        numeric_df = df.select_dtypes(include="number").dropna()

        target_col = "Total Streams (Millions)"
        if target_col not in numeric_df.columns:
            raise ValueError(f"No existe la columna '{target_col}' en el dataset.")

        X = numeric_df.drop(columns=[target_col])
        y = numeric_df[target_col]
        feature_names = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = DecisionTreeRegressor(max_depth=6, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)

        # ==== Gráfico 1: Real vs Predicho ====
        plt.figure(figsize=(7, 5))
        plt.scatter(y_test, y_pred, alpha=0.4)
        min_v, max_v = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        plt.plot([min_v, max_v], [min_v, max_v], "r--", label="Línea ideal")
        plt.xlabel("Streams reales (Millions)")
        plt.ylabel("Streams predichos (Millions)")
        plt.title("Árbol de Decisión – Real vs Predicho")
        plt.legend()
        img_real_pred = os.path.join(STATIC_IMG_DIR, "arbol_real_vs_pred.png")
        plt.savefig(img_real_pred, dpi=120)
        plt.close()

        # ==== Importancia de variables ====
        importances = model.feature_importances_
        idx_sorted = np.argsort(importances)[::-1][:10]
        top_features = [feature_names[i] for i in idx_sorted]
        top_importances = importances[idx_sorted]

        plt.figure(figsize=(8, 5))
        plt.barh(top_features[::-1], top_importances[::-1])
        plt.xlabel("Importancia")
        plt.title("Árbol de Decisión – Importancia de Variables")
        img_importancia = os.path.join(STATIC_IMG_DIR, "arbol_importancia.png")
        plt.tight_layout()
        plt.savefig(img_importancia, dpi=120)
        plt.close()

        importancia_vars = [
            {"variable": f, "importancia": float(imp)}
            for f, imp in zip(top_features, top_importances)
        ]

        return {
            "ok": True,
            "tipo": "Árbol de Decisión",
            "columna_objetivo": target_col,
            "MAE": round(float(mae), 3),
            "MSE": round(float(mse), 3),
            "RMSE": round(float(rmse), 3),
            "R2": round(float(r2), 3),
            "Importancia_Variables": importancia_vars,
            "images": {
                "real_vs_pred": "img/arbol_real_vs_pred.png",
                "importancia": "img/arbol_importancia.png"
            }
        }

    except Exception as e:
        return {
            "ok": False,
            "tipo": "Árbol de Decisión",
            "error": str(e),
            "Importancia_Variables": [],
            "images": {}
        }