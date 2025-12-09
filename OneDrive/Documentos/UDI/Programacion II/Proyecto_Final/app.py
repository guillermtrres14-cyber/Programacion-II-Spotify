from flask import Flask, render_template
import os
import pandas as pd

from modelos.regresion import run_regression_model
from modelos.arbol import run_arbol
from modelos.k_means import run_kmeans
from modelos.sentimientos import run_sentimiento

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Spotify_2024_Global_Streaming_Data.csv")


@app.route("/")
def index():
    # Página de bienvenida / overview
    return render_template("index.html")


@app.route("/regresion")
def vista_regresion():
    results = run_regression_model()        # si tu función recibe ruta, usa run_regression_model(DATA_PATH)
    return render_template("regresion.html", results=results)


@app.route("/arbol")
def vista_arbol():
    metrics = run_arbol(DATA_PATH)
    return render_template("arbol.html", metrics=metrics)


@app.route("/kmeans")
def vista_kmeans():
    metrics = run_kmeans(DATA_PATH)
    return render_template("kmeans.html", metrics=metrics)


@app.route("/")
def index():
    return render_template(
        "index.html",
        selected_model=None,
        metrics=None,
        results=None,
        sentimiento=None
    )


@app.route("/dataset")
def vista_dataset():
    df = pd.read_csv(DATA_PATH, encoding="latin-1")
    tabla = df.head(50).to_html(classes="data-table", index=False, border=0)
    return render_template(
        "dataset.html",
        tabla=tabla,
        n_filas=df.shape[0],
        n_columnas=df.shape[1],
    )


if __name__ == "__main__":
    app.run(debug=True)