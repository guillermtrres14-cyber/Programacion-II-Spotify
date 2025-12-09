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
    csv_path = os.path.join("data", "spotify_reviews.csv")
    sentimiento = run_sentimiento(csv_path)
    return render_template(
        "index.html",
        selected_model="Sentimiento",
        sentimiento=sentimiento,
        metrics=None,
        results=None,
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