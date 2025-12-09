import base64
from io import BytesIO

from flask import Flask, render_template
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sin interfaz gráfica
import matplotlib.pyplot as plt

import numpy as np
import re
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.cluster import KMeans

from wordcloud import WordCloud
import nltk

# ============================================================
# CONFIGURACIÓN DE RUTAS DE ARCHIVOS
# ============================================================

# CSV principal de Spotify (ajusta la ruta si lo tienes en /data/)
RUTA_DATA_SPOTIFY = "data/Spotify_2024_Global_Streaming_Data.csv"

# Columnas reales según tu dataset
COL_Y  = "Total Streams (Millions)"          # objetivo
COL_X1 = "Monthly Listeners (Millions)"      # feature 1
COL_X2 = "Streams Last 30 Days (Millions)"   # feature 2

# CSV de reseñas de la app Spotify (score + content)
RUTA_DATA_REVIEWS = "data/spotify_reviews.csv"


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def fig_to_base64(fig) -> str:
    """Convierte una figura matplotlib a base64 para incrustarla en <img>."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_png = buf.getvalue()
    buf.close()
    return base64.b64encode(img_png).decode("utf-8")


def cargar_spotify() -> pd.DataFrame:
    """Carga el dataset de Spotify y elimina filas con NaN."""
    df = pd.read_csv(RUTA_DATA_SPOTIFY)
    df = df.dropna()
    return df


def cargar_stopwords():
    """Carga las stopwords de NLTK de forma segura (las descarga si no existen)."""
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

    from nltk.corpus import stopwords

    stop_es = set(stopwords.words("spanish"))
    stop_extra = {"spotify", "app", "aplicación", "funciona", "función"}
    return stop_es.union(stop_extra)


STOPWORDS = cargar_stopwords()

# ============================================================
# MODELOS Y GRÁFICAS – REGRESIÓN LINEAL
# ============================================================

def generar_graficas_regresion():
    df = cargar_spotify()

    # ====== Features base para el modelo ======
    # X: oyentes mensuales, y: streams totales
    X = df[[COL_X1]]
    y = df[COL_Y]

    # Feature derivada para "intensidad de escucha"
    df["Streams per Listener"] = df[COL_Y] / df[COL_X1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    imagenes = {}

    # 1) Oyentes mensuales vs Streams (scatter simple)
    fig_a, ax_a = plt.subplots(figsize=(6, 4))
    ax_a.scatter(df[COL_X1], df[COL_Y], alpha=0.5)
    ax_a.set_xlabel("Oyentes mensuales (Monthly Listeners)")
    ax_a.set_ylabel("Streams totales")
    ax_a.set_title("Oyentes mensuales vs Streams")
    imagenes["scatter_listeners"] = fig_to_base64(fig_a)
    plt.close(fig_a)

    # 2) Intensidad de escucha vs Streams (Streams per Listener)
    fig_b, ax_b = plt.subplots(figsize=(6, 4))
    ax_b.scatter(df["Streams per Listener"], df[COL_Y], alpha=0.5)
    ax_b.set_xlabel("Streams per Listener (intensidad de escucha)")
    ax_b.set_ylabel("Streams totales")
    ax_b.set_title("Intensidad de escucha vs Streams")
    imagenes["scatter_intensidad"] = fig_to_base64(fig_b)
    plt.close(fig_b)

    # 3) Scatter + recta de regresión (sobre el conjunto de prueba)
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.scatter(X_test[COL_X1], y_test, alpha=0.5, label="Real")
    ax1.plot(X_test[COL_X1], y_pred, color="red", linewidth=2, label="Predicción")
    ax1.set_xlabel(COL_X1)
    ax1.set_ylabel(COL_Y)
    ax1.set_title(f"{COL_Y} vs {COL_X1} (Regresión)")
    ax1.legend()
    imagenes["modelo"] = fig_to_base64(fig1)
    plt.close(fig1)

    # 4) Real vs Predicho
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.scatter(y_test, y_pred, alpha=0.5)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    ax2.set_xlabel("Real")
    ax2.set_ylabel("Predicho")
    ax2.set_title("Real vs Predicho – Regresión Lineal")
    imagenes["real_vs_pred"] = fig_to_base64(fig2)
    plt.close(fig2)

    # 5) Histograma de residuos
    residuos = y_test - y_pred
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.hist(residuos, bins=30)
    ax3.set_title("Distribución de residuos")
    ax3.set_xlabel("Error (real - predicho)")
    ax3.set_ylabel("Frecuencia")
    imagenes["residuos"] = fig_to_base64(fig3)
    plt.close(fig3)

    metricas = {
        "mse": round(mse, 3),
        "mae": round(mae, 3),
        "r2":  round(r2, 4),
    }

    return imagenes, metricas

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    imagenes = {}

    # 1) Scatter + recta de regresión
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.scatter(X_test[COL_X1], y_test, alpha=0.5, label="Real")
    ax1.plot(X_test[COL_X1], y_pred, color="red", linewidth=2, label="Predicción")
    ax1.set_xlabel(COL_X1)
    ax1.set_ylabel(COL_Y)
    ax1.set_title(f"{COL_Y} vs {COL_X1}")
    ax1.legend()
    imagenes["modelo"] = fig_to_base64(fig1)
    plt.close(fig1)

    # 2) Real vs Predicho
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.scatter(y_test, y_pred, alpha=0.5)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    ax2.set_xlabel("Real")
    ax2.set_ylabel("Predicho")
    ax2.set_title("Real vs Predicho – Regresión Lineal")
    imagenes["real_vs_pred"] = fig_to_base64(fig2)
    plt.close(fig2)

    # 3) Histograma de residuos
    residuos = y_test - y_pred
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.hist(residuos, bins=30)
    ax3.set_title("Distribución de residuos")
    ax3.set_xlabel("Error (real - predicho)")
    ax3.set_ylabel("Frecuencia")
    imagenes["residuos"] = fig_to_base64(fig3)
    plt.close(fig3)

    metricas = {
        "mse": round(mse, 3),
        "mae": round(mae, 3),
        "r2": round(r2, 4),
    }

    return imagenes, metricas


# ============================================================
# MODELOS Y GRÁFICAS – ÁRBOL DE DECISIÓN
# ============================================================

def generar_graficas_arbol():
    df = cargar_spotify()

    X = df[[COL_X1, COL_X2]]
    y = df[COL_Y]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    arbol = DecisionTreeRegressor(max_depth=4, random_state=42)
    arbol.fit(X_train, y_train)

    y_pred = arbol.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    imagenes = {}

    # 1) Árbol de decisión
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    plot_tree(
        arbol,
        feature_names=[COL_X1, COL_X2],
        filled=True,
        rounded=True,
        fontsize=6,
        ax=ax1,
    )
    ax1.set_title("Árbol de decisión – Predicción de streams")
    imagenes["arbol"] = fig_to_base64(fig1)
    plt.close(fig1)

    # 2) Importancia de variables
    importancias = arbol.feature_importances_
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar([COL_X1, COL_X2], importancias)
    ax2.set_title("Importancia de variables")
    ax2.set_ylabel("Importancia")
    imagenes["importancias"] = fig_to_base64(fig2)
    plt.close(fig2)

    metricas = {
        "r2": round(r2, 4),
        "mae": round(mae, 3),
        "profundidad": arbol.get_depth(),
        "n_nodos": arbol.tree_.node_count,
    }

    return imagenes, metricas


# ============================================================
# MODELOS Y GRÁFICAS – K-MEANS
# ============================================================

def generar_graficas_kmeans():
    df = cargar_spotify()

    X = df[[COL_X1, COL_X2]]

    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    imagenes = {}

    # 1) Scatter de clusters
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.scatter(X[COL_X1], X[COL_X2], c=clusters, alpha=0.6)
    centers = kmeans.cluster_centers_
    ax1.scatter(centers[:, 0], centers[:, 1], s=200, marker="X")
    ax1.set_xlabel(COL_X1)
    ax1.set_ylabel(COL_X2)
    ax1.set_title("Clusters K-means")
    imagenes["clusters"] = fig_to_base64(fig1)
    plt.close(fig1)

    # 2) Tamaño de clusters
    sizes = np.bincount(clusters, minlength=k)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(range(k), sizes)
    ax2.set_xlabel("Cluster")
    ax2.set_ylabel("Número de elementos")
    ax2.set_title("Tamaño de cada cluster")
    imagenes["tamanos"] = fig_to_base64(fig2)
    plt.close(fig2)

    metricas = {
        "k": k,
        "inercia": round(kmeans.inertia_, 2),
    }

    return imagenes, metricas


# ============================================================
# DASHBOARD DE SENTIMIENTO (score + texto)
# ============================================================

def generar_graficas_sentimiento():
    df = pd.read_csv(RUTA_DATA_REVIEWS)

    if "content" not in df.columns or "score" not in df.columns:
        raise ValueError(
            "El CSV de reseñas debe tener columnas 'content' y 'score'. "
            f"Columnas actuales: {', '.join(df.columns)}"
        )

    imagenes = {}

    # 1) Distribución de score
    conteo_score = df["score"].value_counts().sort_index()
    fig1, ax1 = plt.subplots()
    conteo_score.plot(kind="bar", ax=ax1)
    ax1.set_xlabel("score")
    ax1.set_ylabel("Número de reseñas")
    ax1.set_title("Distribución de score en reseñas de Spotify")
    imagenes["score"] = fig_to_base64(fig1)
    plt.close(fig1)

    # 2) Longitud de comentarios
    longitudes = df["content"].dropna().astype(str).apply(lambda x: len(x.split()))
    fig2, ax2 = plt.subplots()
    ax2.hist(longitudes, bins=30, color="#3498db", alpha=0.85)
    ax2.set_title("Distribución de longitud de comentarios")
    ax2.set_xlabel("Número de palabras")
    ax2.set_ylabel("Cantidad de reseñas")
    imagenes["longitud"] = fig_to_base64(fig2)
    plt.close(fig2)

    # 3) Top palabras (sin stopwords)
    textos = df["content"].dropna().astype(str)
    todas = []
    for t in textos:
        t = t.lower()
        t = re.sub(r"[^a-záéíóúñü ]", " ", t)
        palabras = [p for p in t.split() if p not in STOPWORDS and len(p) > 3]
        todas.extend(palabras)

    if todas:
        top = Counter(todas).most_common(15)
        palabras_top, frecs = zip(*top)
        fig3, ax3 = plt.subplots(figsize=(9, 6))
        ax3.barh(palabras_top[::-1], frecs[::-1])
        ax3.set_title("Top 15 palabras más mencionadas")
        ax3.set_xlabel("Frecuencia")
        imagenes["top_palabras"] = fig_to_base64(fig3)
        plt.close(fig3)
    else:
        imagenes["top_palabras"] = None

    # 4) WordCloud
    texto_total = " ".join(textos).lower()
    texto_total = re.sub(r"[^a-záéíóúñü ]", " ", texto_total)
    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        colormap="viridis",
        max_words=200,
    ).generate(texto_total)

    fig4 = plt.figure(figsize=(12, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    imagenes["wordcloud"] = fig_to_base64(fig4)
    plt.close(fig4)

    return imagenes


# ============================================================
# RUTAS FLASK
# ============================================================

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/regresion")
def vista_regresion():
    try:
        imagenes, metricas = generar_graficas_regresion()
        error = None
    except Exception as e:
        imagenes = metricas = None
        error = str(e)

    return render_template(
        "regresion.html",
        imagenes=imagenes,
        metricas=metricas,
        error=error,
    )


@app.route("/arbol")
def vista_arbol():
    try:
        imagenes, metricas = generar_graficas_arbol()
        error = None
    except Exception as e:
        imagenes = metricas = None
        error = str(e)

    return render_template(
        "arbol.html",
        imagenes=imagenes,
        metricas=metricas,
        error=error,
    )


@app.route("/kmeans")
def vista_kmeans():
    try:
        imagenes, metricas = generar_graficas_kmeans()
        error = None
    except Exception as e:
        imagenes = metricas = None
        error = str(e)

    return render_template(
        "kmeans.html",
        imagenes=imagenes,
        metricas=metricas,
        error=error,
    )


@app.route("/sentimiento")
def vista_sentimiento():
    try:
        imagenes = generar_graficas_sentimiento()
        error = None
    except Exception as e:
        imagenes = None
        error = str(e)

    return render_template(
        "sentimiento.html",
        imagenes=imagenes,
        error=error,
    )


@app.route("/dataset")
def vista_dataset():
    try:
        df = cargar_spotify()
        filas, columnas = df.shape

        tabla_html = df.head(50).to_html(
            classes="table table-striped table-sm",
            index=False
        )
        error = None
    except Exception as e:
        tabla_html = None
        filas = columnas = 0
        error = str(e)

    return render_template(
        "dataset.html",
        tabla_html=tabla_html,
        filas=filas,
        columnas=columnas,
        error=error
    )
# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)