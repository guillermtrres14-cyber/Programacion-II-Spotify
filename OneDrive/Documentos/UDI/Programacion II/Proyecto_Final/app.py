# app.py
import base64
from io import BytesIO

from flask import Flask, render_template
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sin ventana gráfica
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.cluster import KMeans

app = Flask(__name__)

# --------------------------------------------------------------------
# CONFIGURACIÓN – AJUSTA ESTO A TUS ARCHIVOS Y COLUMNAS
# --------------------------------------------------------------------
# CSV principal de Spotify (streams + features)
RUTA_DATA_SPOTIFY = "data/Spotify_2024_Global_Streaming_Data.csv"

# nombres de columnas QUE EXISTEN en tu CSV principal
COLUMNA_STREAMS = "streams"        # columna objetivo
COLUMNA_FEATURE_X = "danceability" # input para regresión
COLUMNA_FEATURE_X2 = "energy"      # segunda feature (árbol y k-means)

# CSV de reseñas con sentimientos
RUTA_DATA_REVIEWS = "data/spotify_reviews.csv"
COLUMNA_SENTIMIENTO = "sentiment"  # positive / neutral / negative, etc.


# --------------------------------------------------------------------
# FUNCIONES AUXILIARES
# --------------------------------------------------------------------
def df_spotify():
    """Carga el dataset principal de Spotify."""
    return pd.read_csv(RUTA_DATA_SPOTIFY)


def fig_to_base64(fig):
    """Convierte una figura matplotlib a base64 para incrustarla en <img>."""
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    img_png = buffer.getvalue()
    buffer.close()
    return base64.b64encode(img_png).decode("utf-8")


# ------------------------- REGRESIÓN LINEAL ------------------------- #
def grafico_regresion_y_metricas():
    df = df_spotify()

    X = df[[COLUMNA_FEATURE_X]]
    y = df[COLUMNA_STREAMS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    r2 = modelo.score(X_test, y_test)

    fig, ax = plt.subplots()
    ax.scatter(X_test, y_test, alpha=0.5, label="Real")
    ax.plot(X_test, y_pred, linewidth=2, label="Predicción")
    ax.set_xlabel(COLUMNA_FEATURE_X.capitalize())
    ax.set_ylabel("Streams")
    ax.set_title("Regresión lineal: Streams vs " + COLUMNA_FEATURE_X)
    ax.legend()

    img64 = fig_to_base64(fig)
    plt.close(fig)

    metricas = {"r2": round(r2, 4)}
    return img64, metricas


# ------------------------ ÁRBOL DE DECISIÓN ------------------------ #
def grafico_arbol_y_metricas():
    df = df_spotify()

    X = df[[COLUMNA_FEATURE_X, COLUMNA_FEATURE_X2]]
    y = df[COLUMNA_STREAMS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    arbol = DecisionTreeRegressor(max_depth=4, random_state=42)
    arbol.fit(X_train, y_train)

    score = arbol.score(X_test, y_test)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_tree(
        arbol,
        feature_names=[COLUMNA_FEATURE_X, COLUMNA_FEATURE_X2],
        filled=True,
        rounded=True,
        fontsize=6,
        ax=ax
    )
    ax.set_title("Árbol de decisión - Predicción de streams")

    img64 = fig_to_base64(fig)
    plt.close(fig)

    metricas = {"r2": round(score, 4)}
    return img64, metricas


# ----------------------------- K-MEANS ------------------------------ #
def grafico_kmeans():
    df = df_spotify()
    X = df[[COLUMNA_FEATURE_X, COLUMNA_FEATURE_X2]]

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    fig, ax = plt.subplots()
    scatter = ax.scatter(
        X[COLUMNA_FEATURE_X],
        X[COLUMNA_FEATURE_X2],
        c=clusters,
        alpha=0.6
    )
    centers = kmeans.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], s=200, marker="X")

    ax.set_xlabel(COLUMNA_FEATURE_X.capitalize())
    ax.set_ylabel(COLUMNA_FEATURE_X2.capitalize())
    ax.set_title("Clusters K-means")

    img64 = fig_to_base64(fig)
    plt.close(fig)

    return img64


# ------------------------ ANÁLISIS DE SENTIMIENTO ------------------- #
def grafico_sentimiento():
    """
    Lee un CSV de reseñas con una columna de sentimiento
    y genera un gráfico de barras con la distribución.
    """
    try:
        df = pd.read_csv(RUTA_DATA_REVIEWS)
    except Exception:
        # si no existe el archivo o hay error, devolvemos None
        return None

    conteo = df[COLUMNA_SENTIMIENTO].value_counts()

    fig, ax = plt.subplots()
    conteo.plot(kind="bar", ax=ax)
    ax.set_xlabel("Sentimiento")
    ax.set_ylabel("Número de reseñas")
    ax.set_title("Distribución de sentimiento en reseñas de Spotify")

    img64 = fig_to_base64(fig)
    plt.close(fig)

    return img64


# --------------------------------------------------------------------
# RUTAS FLASK
# --------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/regresion")
def vista_regresion():
    try:
        grafico, metricas = grafico_regresion_y_metricas()
        error = None
    except Exception as e:
        grafico = None
        metricas = None
        error = str(e)

    return render_template(
        "regresion.html",
        grafico_regresion=grafico,
        metricas=metricas,
        error=error
    )


@app.route("/arbol")
def vista_arbol():
    try:
        grafico, metricas = grafico_arbol_y_metricas()
        error = None
    except Exception as e:
        grafico = None
        metricas = None
        error = str(e)

    return render_template(
        "arbol.html",
        grafico_arbol=grafico,
        metricas=metricas,
        error=error
    )


@app.route("/kmeans")
def vista_kmeans():
    try:
        grafico = grafico_kmeans()
        error = None
    except Exception as e:
        grafico = None
        error = str(e)

    return render_template(
        "kmeans.html",
        grafico_kmeans=grafico,
        error=error
    )


@app.route("/sentimiento")
def vista_sentimiento():
    grafico = grafico_sentimiento()
    error = None
    if grafico is None:
        error = "No se encontró el archivo de reseñas o la columna de sentimiento."

    return render_template(
        "sentimiento.html",
        grafico_sentimiento=grafico,
        error=error
    )


@app.route("/dataset")
def vista_dataset():
    """
    Vista del dataset para que el enlace del menú no falle.
    Muestra las primeras filas de la tabla.
    """
    try:
        df = df_spotify()
        tabla_html = df.head(50).to_html(
            classes="table table-striped table-sm",
            index=False
        )
        error = None
    except Exception as e:
        tabla_html = None
        error = str(e)

    return render_template(
        "dataset.html",
        tabla_html=tabla_html,
        error=error
    )


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)