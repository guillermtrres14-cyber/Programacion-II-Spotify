from flask import Flask, render_template, redirect, url_for
import pandas as pd
import os

# Importar modelos
from modelos.regresion import run_regression_model
from modelos.arbol import run_arbol
from modelos.k_means import run_kmeans

app = Flask(__name__)

# Ruta base del proyecto y dataset de Spotify
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Spotify_2024_Global_Streaming_Data.csv")


# ============================
#       RUTA PRINCIPAL
# ============================
@app.route("/")
def index():
    """
    Página principal del dashboard.
    Ejecuta la regresión lineal y envía los resultados al index.
    """
    results = run_regression_model()
    return render_template(
        "index.html",
        selected_model="Regresión Lineal",
        results=results
    )


# ============================
#   VISTA EXPLÍCITA REGRESIÓN
#   (para el link vista_regresion del index.html)
# ============================
@app.route("/regresion")
def vista_regresion():
    """
    Endpoint usado por url_for('vista_regresion') en index.html.
    Puedes redirigir a la página principal o volver a ejecutar la regresión.
    Aquí simplemente redirigimos al index.
    """
    return redirect(url_for("index"))


# ============================
#     ÁRBOL DE DECISIÓN
# ============================
@app.route("/arbol")
def vista_arbol():
    """
    Ejecuta el modelo de Árbol de Decisión y muestra sus métricas
    reutilizando la plantilla index.html.
    """
    metrics = run_arbol(DATA_PATH)
    return render_template(
        "index.html",
        selected_model="Árbol de Decisión",
        metrics=metrics
    )


# ============================
#         K-MEANS
# ============================
@app.route("/kmeans")
def vista_kmeans():
    """
    Ejecuta el modelo K-means y muestra sus métricas en la misma plantilla.
    """
    metrics = run_kmeans(DATA_PATH)
    return render_template(
        "index.html",
        selected_model="K-means",
        metrics=metrics
    )


# ============================
#          DATASET
# ============================
@app.route("/dataset")
def vista_dataset():
    """
    Muestra una tabla con las primeras filas del dataset original.
    """
    df = pd.read_csv(DATA_PATH, encoding="latin-1")

    tabla = df.head(50).to_html(
        classes="data-table",
        index=False,
        border=0
    )

    return render_template(
        "dataset.html",
        tabla=tabla,
        n_filas=df.shape[0],
        n_columnas=df.shape[1]
    )


# ============================
#            MAIN
# ============================
if __name__ == "__main__":
    app.run(debug=True)