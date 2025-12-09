from flask import Flask, render_template
import pandas as pd

# Importar funciones de tus modelos
from modelos.regresion import run_regression_model
from modelos.arbol import run_arbol
from modelos.k_means import run_kmeans
from modelos.sentimientos import run_sentimiento

app = Flask(__name__)

# Rutas de los datasets
DATA_STREAM_PATH = "data/Spotify_2024_Global_Streaming_Data.csv"
REVIEWS_PATH = "data/spotify_reviews.csv"


# ------------------ PÁGINA PRINCIPAL ------------------ #
@app.route("/")
def index():
    """
    Portada del dashboard (solo texto y botones).
    No ejecuta modelos pesados aquí.
    """
    return render_template(
        "index.html",
        selected_model=None,
        metrics=None,
        sentimiento=None
    )


# ------------------ REGRESIÓN LINEAL ------------------ #
@app.route("/regresion")
def vista_regresion():
    """
    Página de regresión lineal.
    """
    results = run_regression_model(DATA_STREAM_PATH)

    return render_template(
        "regresion.html",
        selected_model="Regresión Lineal",
        results=results
    )


# ------------------ ÁRBOL DE DECISIÓN ------------------ #
@app.route("/arbol")
def vista_arbol():
    """
    Página de árbol de decisión.
    """
    metrics = run_arbol(DATA_STREAM_PATH)

    return render_template(
        "arbol.html",
        selected_model="Árbol de Decisión",
        metrics=metrics
    )


# ------------------ K-MEANS CLUSTERING ------------------ #
@app.route("/kmeans")
def vista_kmeans():
    """
    Página de clustering K-means.
    """
    metrics = run_kmeans(DATA_STREAM_PATH)

    return render_template(
        "kmeans.html",
        selected_model="K-means",
        metrics=metrics
    )


# ------------------ ANÁLISIS DE SENTIMIENTO ------------------ #
@app.route("/sentimiento")
def vista_sentimiento():
    """
    Página de análisis de sentimiento usando spotify_reviews.csv
    """
    sentimiento = run_sentimiento(REVIEWS_PATH)

    return render_template(
        "sentimiento.html",
        sentimiento=sentimiento
    )


# ------------------ VISTA DEL DATASET ------------------ #
@app.route("/dataset")
def vista_dataset():
    """
    Muestra una tabla con una muestra del dataset principal.
    """
    df = pd.read_csv(DATA_STREAM_PATH, encoding="latin-1")
    sample = df.head(100).to_dict(orient="records")
    columns = df.columns.tolist()

    return render_template(
        "dataset.html",
        columns=columns,
        rows=sample
    )


# ------------------ MAIN ------------------ #
if __name__ == "__main__":
    app.run(debug=True)
Con este app.py:

Desde tu terminal de VS Code (estando en la carpeta del proyecto y con el venv activo):

powershell
Copiar código
(venv) PS C:\Users\...\Proyecto_Final> python app.py
En el navegador:

http://127.0.0.1:5000/ → portada (index).

http://127.0.0.1:5000/regresion

http://127.0.0.1:5000/arbol

http://127.0.0.1:5000/kmeans

http://127.0.0.1:5000/sentimiento

http://127.0.0.1:5000/dataset

En tus plantillas asegúrate de que los enlaces usan estos nombres de función, por ejemplo en base.html:

html
Copiar código
<a href="{{ url_for('index') }}">Inicio</a>
<a href="{{ url_for('vista_regresion') }}">Regresión Lineal</a>
<a href="{{ url_for('vista_arbol') }}">Árbol de Decisión</a>
<a href="{{ url_for('vista_kmeans') }}">K-means</a>
<a href="{{ url_for('vista_sentimiento') }}">Sentimiento</a>
<a href="{{ url_for('vista_dataset') }}">Dataset</a>