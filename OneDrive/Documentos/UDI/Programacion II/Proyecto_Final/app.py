from flask import Flask, render_template
import pandas as pd
import os

# Importar modelos
from modelos.regresion_model import run_regression_model
from modelos.arbol import run_arbol
from modelos.k_means import run_kmeans

app = Flask(__name__)

# Ruta al dataset Spotify
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Spotify_2024_Global_Streaming_Data.csv")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/")
def index():
    results = run_regression_model()
    return render_template("index.html", results=results)


@app.route("/arbol")
def vista_arbol():
    metrics = run_arbol(DATA_PATH)
    return render_template("index.html", selected_model="Árbol de Decisión", metrics=metrics)


@app.route("/kmeans")
def vista_kmeans():
    metrics = run_kmeans(DATA_PATH)
    return render_template("index.html", selected_model="K-means", metrics=metrics)


@app.route("/dataset")
def vista_dataset():
    df = pd.read_csv(DATA_PATH, encoding="latin-1")
    tabla = df.head(50).to_html(classes="data-table", index=False, border=0)
    return render_template("dataset.html",
                           tabla=tabla,
                           n_filas=df.shape[0],
                           n_columnas=df.shape[1])


if __name__ == "__main__":
    app.run(debug=True)