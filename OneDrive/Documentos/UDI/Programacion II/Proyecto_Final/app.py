from flask import Flask, render_template
import os
import pandas as pd

from modelos.regresion import run_regression_model
from modelos.arbol import run_arbol
from modelos.k_means import run_kmeans
from modelos.sentimientos import run_sentimiento

app = Flask(__name__)

# =========================
# RUTAS DE ARCHIVOS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "Spotify_2024_Global_Streaming_Data.csv")
REVIEWS_FILE = os.path.join(BASE_DIR, "data", "spotify_reviews.csv")


# =========================
# HELPER: VISTA DEL DATASET
# =========================
def load_dataset_head(n: int = 15) -> str | None:
    """
    Devuelve un HTML con las primeras filas del dataset principal.
    Si hay un error, devuelve None.
    """
    try:
        df = pd.read_csv(DATA_FILE, encoding="latin-1")
        return df.head(n).to_html(
            classes="dataset-table",
            index=False,
            border=0
        )
    except Exception:
        return None


# =========================
# RUTA PRINCIPAL
# =========================
@app.route("/")
def index():
    """
    Landing: sin modelo seleccionado, sólo el layout.
    """
    return render_template(
        "index.html",
        selected_model=None,
        metrics=None,
        results=None,
        sentimiento=None,
        dataset_head=None,
    )


# =========================
# REGRESIÓN LINEAL
# =========================
@app.route("/regresion")
def vista_regresion():
    """
    Ejecuta el modelo de Regresión Lineal y muestra sus métricas y gráficas.
    """
    resultados = run_regresion(DATA_FILE)

    return render_template(
        "index.html",
        selected_model="Regresión Lineal",
        metrics=resultados,   # para R2, MSE, MAE, etc.
        results=resultados,   # para acceder a results.images.scatter, etc.
        sentimiento=None,
        dataset_head=None,
    )


# =========================
# ÁRBOL DE DECISIÓN
# =========================
@app.route("/arbol")
def vista_arbol():
    """
    Ejecuta el Árbol de Decisión y muestra métricas y gráficas.
    """
    metrics = run_arbol(DATA_FILE)

    return render_template(
        "index.html",
        selected_model="Árbol de Decisión",
        metrics=metrics,      # contiene métricas + metrics.images.*
        results=None,
        sentimiento=None,
        dataset_head=None,
    )


# =========================
# K-MEANS
# =========================
@app.route("/kmeans")
def vista_kmeans():
    """
    Ejecuta K-means y muestra los clusters (PCA + tamaños).
    """
    metrics = run_kmeans(DATA_FILE)

    return render_template(
        "index.html",
        selected_model="K-means",
        metrics=metrics,      # contiene n_clusters, inercia, metrics.images.*
        results=None,
        sentimiento=None,
        dataset_head=None,
    )


# =========================
# ANÁLISIS DE SENTIMIENTO
# =========================
@app.route("/sentimiento")
def vista_sentimiento():
    """
    Ejecuta el análisis de sentimiento usando spotify_reviews.csv
    y muestra las gráficas generadas.
    """
    sentimiento = run_sentimiento(REVIEWS_FILE)

    return render_template(
        "index.html",
        selected_model="Análisis de Sentimiento",
        metrics=None,
        results=None,
        sentimiento=sentimiento,  # dict con ok, images.*, error si falla
        dataset_head=None,
    )


# =========================
# VISTA DEL DATASET
# =========================
@app.route("/dataset")
def vista_dataset():
    """
    Muestra una tabla con una muestra del dataset principal.
    """
    head_html = load_dataset_head(20)

    return render_template(
        "index.html",
        selected_model="Dataset",
        metrics=None,
        results=None,
        sentimiento=None,
        dataset_head=head_html,
    )


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # Ejecutar siempre desde la carpeta del proyecto:
    # (venv) PS C:\Users\...\Proyecto_Final> python app.py
    app.run(debug=True)