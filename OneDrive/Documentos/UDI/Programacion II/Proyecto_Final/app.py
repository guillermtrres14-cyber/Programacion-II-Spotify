# app.py
from flask import Flask, render_template, url_for

app = Flask(__name__)


# ---------------- RUTAS PRINCIPALES ---------------- #

@app.route("/")
def index():
    """
    Página de inicio / dashboard general.
    """
    return render_template("index.html")


@app.route("/regresion")
def vista_regresion():
    """
    Vista de resultados del modelo de Regresión Lineal.
    Muestra la gráfica guardada como imagen estática.
    """
    img_path = url_for("static", filename="img/regresion_streams.png")
    return render_template("regresion.html", img_regresion=img_path)


@app.route("/arbol")
def vista_arbol():
    """
    Vista del Árbol de Decisión.
    """
    img_path = url_for("static", filename="img/arbol_decision.png")
    return render_template("arbol.html", img_arbol=img_path)


@app.route("/kmeans")
def vista_kmeans():
    """
    Vista del clustering K-means.
    """
    img_path = url_for("static", filename="img/kmeans_clusters.png")
    return render_template("kmeans.html", img_kmeans=img_path)


@app.route("/sentimiento")
def vista_sentimiento():
    """
    Vista del análisis de sentimiento de reseñas.
    """
    img_path = url_for("static", filename="img/sentimiento_reviews.png")
    return render_template("sentimiento.html", img_sentimiento=img_path)


@app.route("/dataset")
def vista_dataset():
    """
    Vista informativa del dataset (si tienes una página para esto).
    """
    return render_template("dataset.html")


# ---------------- ARRANQUE DE LA APP ---------------- #

if __name__ == "__main__":
    app.run(debug=True)