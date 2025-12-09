from flask import Flask, render_template, redirect, url_for
import pandas as pd
import os

# Importar modelos corregidos
from modelos.regresion_model import run_regression_model
from modelos.arbol import run_arbol
from modelos.k_means import run_kmeans

app = Flask(__name__)

# Ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Spotify_2024_Global_Streaming_Data.csv")


# ============================
#     RUTA PRINCIPAL
# ============================
@app.route("/")
def index():
    """Página principal con resultados de regresión."""
    results = run_regression_model()  # Ejecuta regresión automáticamente
    return render_template("index.html",
                           selected_model="Regresión Lineal",
                           results=results)


# ============================
#     ÁRBOL DE DECISIÓN
# ============================
@app.route("/arbol")
def vista_arbol():
    """Página que muestra resultados del árbol de decisión."""
    metrics = run_arbol(DATA_PATH)
    return render_template("index.html",
                           selected_model="Árbol de Decisión",
                           metrics=metrics)


# ============================
#     K-MEANS
# ============================
@app.route("/kmeans")
def vista_kmeans():
    """Página que muestra resultados de clustering K-Means."""
    metrics = run_kmeans(DATA_PATH)
    return render_template("index.html",
                           selected_model="K-means",
                           metrics=metrics)


# ============================
#     DATASET
# ============================
@app.route("/dataset")
def vista_dataset():
    """Vista para mostrar una parte del dataset original."""
    df = pd.read_csv(DATA_PATH, encoding="latin-1")

    tabla = df.head(50).to_html(
        classes="data-table",
        index=False,
        border=0
    )

    return render_template("dataset.html",
                           tabla=tabla,
                           n_filas=df.shape[0],
                           n_columnas=df.shape[1])


# ============================
#     MAIN
# ============================
if __name__ == "__main__":
    app.run(debug=True)
