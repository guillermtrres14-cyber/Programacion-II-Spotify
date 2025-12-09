import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# === RUTAS DEL PROYECTO ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))   # carpeta PROYECTO_FINAL
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_IMG_DIR = os.path.join(BASE_DIR, "static", "img")

os.makedirs(STATIC_IMG_DIR, exist_ok=True)


def run_regression_model():
    try:
        # === 1) Cargar dataset de Spotify ===
        file_path = os.path.join(DATA_DIR, "Spotify_2024_Global_Streaming_Data.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No se encontró el dataset en: {file_path}")

        df = pd.read_csv(file_path)

        # === 2) Limpiar y seleccionar numéricas ===
        numeric_df = df.select_dtypes(include="number").dropna()

        target = "Total Streams (Millions)"
        if target not in numeric_df.columns:
            raise ValueError(f"La columna objetivo '{target}' no existe en el dataset")

        X = numeric_df.drop(columns=[target])
        y = numeric_df[target]

        # === 3) Train/test ===
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        # === 4) Métricas ===
        mse = mean_squared_error(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)

        # === 5) Mejor variable correlacionada ===
        corr = numeric_df.corr()[target].drop(target)
        best_feature = corr.abs().sort_values(ascending=False).index[0]

        # === 6) Gráfico Scatter + Regresión ===
        X_best = numeric_df[[best_feature]]
        y_all = numeric_df[target]

        line_model = LinearRegression()
        line_model.fit(X_best, y_all)

        x_line = np.linspace(X_best.min(), X_best.max(), 100)
        y_line = line_model.predict(pd.DataFrame({best_feature: x_line.flatten()}))

        plt.figure(figsize=(7, 5))
        plt.scatter(X_best, y_all, alpha=0.3, label="Datos reales")
        plt.plot(x_line, y_line, color="red", linewidth=2, label="Regresión")
        plt.xlabel(best_feature)
        plt.ylabel(target)
        plt.title(f"Relación entre {best_feature} y {target}")
        plt.legend()
        scatter_path = os.path.join(STATIC_IMG_DIR, "regresion_scatter.png")
        plt.savefig(scatter_path, dpi=120)
        plt.close()

        # === 7) Importancia (coeficientes) ===
        coef_importance = np.abs(model.coef_)
        feature_names = list(X.columns)

        plt.figure(figsize=(8, 5))
        plt.barh(feature_names, coef_importance)
        plt.title("Importancia de Variables (|coeficiente|)")
        plt.xlabel("Importancia")
        plt.tight_layout()
        coef_path = os.path.join(STATIC_IMG_DIR, "regresion_importancia.png")
        plt.savefig(coef_path, dpi=120)
        plt.close()

        return {
            "ok": True,
            "target": target,
            "best_feature": best_feature,
            "mse": round(float(mse), 3),
            "mae": round(float(mae), 3),
            "r2": round(float(r2), 3),
            "images": {
                "scatter": "img/regresion_scatter.png",
                "importancia": "img/regresion_importancia.png"
            }
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "images": {}
        }