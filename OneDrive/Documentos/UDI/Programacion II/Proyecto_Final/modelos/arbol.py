import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Rutas del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_IMG_DIR = os.path.join(BASE_DIR, "static", "img")
os.makedirs(STATIC_IMG_DIR, exist_ok=True)


def run_arbol(data_path: str | None = None) -> dict:
    """
    Entrena un Árbol de Decisión para predecir 'Total Streams (Millions)'.
    Devuelve métricas y rutas de imágenes para mostrarlas en la web.
    """

    try:
        # 1. Ruta del dataset
        if data_path is None:
            data_path = os.path.join(DATA_DIR, "Spotify_2024_Global_Streaming_Data.csv")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"No se encontró el dataset en: {data_path}")

        # 2. Cargar datos
        df = pd.read_csv(data_path)

        # 3. Quedarnos con columnas numéricas
        numeric_df = df.select_dtypes(include="number").dropna()

        target_col = "Total Streams (Millions)"
        if target_col not in numeric_df.columns:
            raise ValueError(f"No existe la columna objetivo '{target_col}' en el dataset.")

        X = numeric_df.drop(columns=[target_col])
        y = numeric_df[target_col]
        feature_names = list(X.columns)

        # 4. Train / Test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 5. Modelo Árbol de Decisión
        model = DecisionTreeRegressor(
            random_state=42,
            max_depth=6  # profundidad moderada para no sobreajustar demasiado
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # 6. Métricas
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)

        # 7. Importancia de variables
        importances = model.feature_importances_
        # Top 10 más importantes
        idx_sorted = np.argsort(importances)[::-1][:10]
        top_features = [feature_names[i] for i in idx_sorted]
        top_importances = importances[idx_sorted]

        # === Gráfico 1: Real vs Predicho ===
        plt.figure(figsize=(7, 5))
        plt.scatter(y_test, y_pred, alpha=0.4)
        max_val = max(y_test.max(), y_pred.max())
        min_val = min(y_test.min(), y_pred.min())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Línea ideal")
        plt.xlabel("Total Streams reales (Millions)")
        plt.ylabel("Total Streams predichos (Millions)")
        plt.title("Árbol de Decisión – Real vs Predicho")
        plt.legend()
        plt.tight_layout()
        real_pred_path = os.path.join(STATIC_IMG_DIR, "arbol_real_vs_pred.png")
        plt.savefig(real_pred_path, dpi=120)
        plt.close()

        # === Gráfico 2: Importancia de variables ===
        plt.figure(figsize=(8, 5))
        plt.barh(top_features[::-1], top_importances[::-1])
        plt.xlabel("Importancia")
        plt.title("Árbol de Decisión – Importancia de Variables")
        plt.tight_layout()
        import_path = os.path.join(STATIC_IMG_DIR, "arbol_importancia.png")
        plt.savefig(import_path, dpi=120)
        plt.close()

        # 8. Construir respuesta
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
                "importancia": "img/arbol_importancia.png",
            },
        }

    except Exception as e:
        return {
            "ok": False,
            "tipo": "Árbol de Decisión",
            "error": f"Error al ejecutar el Árbol de Decisión: {str(e)}",
            "Importancia_Variables": [],
            "images": {},
        }
Asegúrate de que en la carpeta modelos/ tengas:

text
Copiar código
modelos/
   __init__.py      (puede estar vacío)
   regresion.py
   arbol.py
   k_means.py
2️⃣ app.py: la ruta /arbol ya sirve, solo úsala así
Tu app.py con las rutas tipo “bajo botón” debería verse así (solo te repito la parte relevante):

python
Copiar código
from modelos.regresion import run_regression_model
from modelos.arbol import run_arbol
from modelos.k_means import run_kmeans
...
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/regresion")
def vista_regresion():
    results = run_regression_model()
    return render_template(
        "index.html",
        selected_model="Regresión Lineal",
        results=results
    )

@app.route("/arbol")
def vista_arbol():
    metrics = run_arbol(DATA_PATH)
    return render_template(
        "index.html",
        selected_model="Árbol de Decisión",
        metrics=metrics
    )