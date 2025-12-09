import os
import pandas as pd
from flask import Flask, render_template

# Importar funciones de los modelos
from modelos.regresion import run_regresion
from modelos.arbol import run_arbol
from modelos.k_means import run_kmeans
from modelos.sentimientos import run_sentimiento

# Rutas base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

app = Flask(__name__)


# -------------------- Helpers -------------------- #

def get_spotify_csv():
    """
    Devuelve la ruta del CSV principal de Spotify para regresión / k-means / árbol.
    Busca un .csv en la carpeta data, priorizando los que contengan 'spotify'
    y que NO sean de reseñas.
    """
    if not os.path.isdir(DATA_DIR):
        return None

    archivos = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")]
    if not archivos:
        return None

    # Preferir dataset grande de streaming (Spotify global)
    for name in archivos:
        low = name.lower()
        if "spotify" in low and "review" not in low:
            return os.path.join(DATA_DIR, name)

    # Si no hay nada más específico, usar el primero
    return os.path.join(DATA_DIR, archivos[0])


def get_reviews_csv():
    """
    Devuelve la ruta del CSV de reseñas para análisis de sentimiento.
    Espera un archivo 'spotify_reviews.csv' en la carpeta data.
    """
    path = os.path.join(DATA_DIR, "spotify_reviews.csv")
    return path if os.path.exists(path) else None


# -------------------- Rutas principales -------------------- #

@app.route("/")
def index():
    # Portada / dashboard principal
    return render_template("index.html")


# ---------- Vista rápida del dataset ---------- #

@app.route("/dataset")
def vista_dataset():
    csv_path = get_spotify_csv()

    if not csv_path or not os.path.exists(csv_path):
        return render_template(
            "dataset.html",
            error="No se encontró ningún archivo CSV en la carpeta data.",
            rows=None,
            cols=None,
            table_html=None,
            csv_name=None,
        )

    df = pd.read_csv(csv_path)

    rows, cols = df.shape
    # Muestra hasta 50 registros
    sample = df.head(50)

    table_html = sample.to_html(
        classes="table table-dark table-striped table-sm",
        index=False,
        border=0,
    )

    return render_template(
        "dataset.html",
        error=None,
        rows=rows,
        cols=cols,
        table_html=table_html,
        csv_name=os.path.basename(csv_path),
    )


# ---------- Regresión Lineal ---------- #

@app.route("/regresion")
def vista_regresion():
    csv_path = get_spotify_csv()

    if not csv_path or not os.path.exists(csv_path):
        return render_template(
            "regresion.html",
            error="No se encontró el dataset de Spotify en la carpeta data.",
            result=None,
            csv_name=None,
        )

    try:
        # IMPORTANTE: respetamos la firma original run_regresion(csv_path)
        result = run_regresion(csv_path)
        error = None
    except Exception as e:
        # Verás el detalle del error en la consola de VS Code
        print("Error en run_regresion:", e)
        result = None
        error = "No se pudieron calcular las métricas de regresión."

    return render_template(
        "regresion.html",
        error=error,
        result=result,
        csv_name=os.path.basename(csv_path),
    )


# ---------- Árbol de Decisión ---------- #

@app.route("/arbol")
def vista_arbol():
    csv_path = get_spotify_csv()

    if not csv_path or not os.path.exists(csv_path):
        return render_template(
            "arbol.html",
            error="No se encontró el dataset de Spotify en la carpeta data.",
            result=None,
        )

    try:
        result = run_arbol(csv_path)
        error = None
    except Exception as e:
        print("Error en run_arbol:", e)
        result = None
        error = "No se pudo entrenar el árbol de decisión."

    return render_template("arbol.html", error=error, result=result)


# ---------- K-means ---------- #

@app.route("/kmeans")
def vista_kmeans():
    csv_path = get_spotify_csv()

    if not csv_path or not os.path.exists(csv_path):
        return render_template(
            "kmeans.html",
            error="No se encontró el dataset de Spotify en la carpeta data.",
            result=None,
        )

    try:
        result = run_kmeans(csv_path)
        error = None
    except Exception as e:
        print("Error en run_kmeans:", e)
        result = None
        error = "No se pudo ejecutar el modelo de K-means."

    return render_template("kmeans.html", error=error, result=result)


# ---------- Análisis de Sentimiento ---------- #

@app.route("/sentimiento")
def vista_sentimiento():
    csv_path = get_reviews_csv()

    if not csv_path:
        return render_template(
            "sentimiento.html",
            error="No se encontró 'spotify_reviews.csv' en la carpeta data.",
            result=None,
        )

    try:
        result = run_sentimiento(csv_path)
        error = None
    except Exception as e:
        print("Error en run_sentimiento:", e)
        result = None
        error = (
            "No se pudo procesar el análisis de sentimiento. "
            "Revisa que el CSV tenga una columna llamada 'review'."
        )

    return render_template("sentimiento.html", error=error, result=result)


# -------------------- Main -------------------- #

if __name__ == "__main__":
    # Ejecutar en modo debug mientras estás desarrollando
    app.run(debug=True)