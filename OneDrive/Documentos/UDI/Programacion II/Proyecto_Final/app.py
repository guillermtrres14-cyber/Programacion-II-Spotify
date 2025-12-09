from wordcloud import WordCloud
from nltk.corpus import stopwords
import re
from collections import Counter
import nltk

# === CONFIGURACIÓN DEL DATASET SPOTIFY (IGUAL A LOS NOTEBOOKS) ===
RUTA_DATA_SPOTIFY = "Spotify_2024_Global_Streaming_Data.csv"

NUMERIC_BASE = [
    "Release Year",
    "Monthly Listeners (Millions)",
    "Total Streams (Millions)",
    "Total Hours Streamed (Millions)",
    "Avg Stream Duration (Min)",
    "Streams Last 30 Days (Millions)",
    "Skip Rate (%)",
]

TARGET_COL = "Total Streams (Millions)"  # variable objetivo


def cargar_spotify_limpio():
    """
    Replica la preparación de datos de tus notebooks:
    - lee el CSV
    - elimina NaN
    - crea Streams per Listener, Recent Streams Ratio, Hours per Listener
    - devuelve df_clean, numeric_cols, feature_cols
    """
    df = pd.read_csv(RUTA_DATA_SPOTIFY)
    df_clean = df.copy().dropna()

    # Nuevas variables de negocio (igual que en tus notebooks)
    df_clean["Streams per Listener"] = (
        df_clean["Total Streams (Millions)"] / df_clean["Monthly Listeners (Millions)"]
    )
    df_clean["Recent Streams Ratio"] = (
        df_clean["Streams Last 30 Days (Millions)"]
        / df_clean["Total Streams (Millions)"]
    )
    df_clean["Hours per Listener"] = (
        df_clean["Total Hours Streamed (Millions)"]
        / df_clean["Monthly Listeners (Millions)"]
    )

    numeric_cols = NUMERIC_BASE + [
        "Streams per Listener",
        "Recent Streams Ratio",
        "Hours per Listener",
    ]
    feature_cols = [c for c in numeric_cols if c != TARGET_COL]

    return df_clean, numeric_cols, feature_cols

# CSV principal de Spotify (tu archivo actual)
RUTA_DATA_SPOTIFY = "data/Spotify_2024_Global_Streaming_Data.csv"

# >>>> NOMBRES REALES SEGÚN TU CSV <<<<
COLUMNA_STREAMS = "Total Streams (Millions)"          # objetivo y
COLUMNA_FEATURE_X = "Monthly Listeners (Millions)"    # X1
COLUMNA_FEATURE_X2 = "Streams Last 30 Days (Millions)"  # X2

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
def cargar_stopwords():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

    from nltk.corpus import stopwords

    STOP_ES = set(stopwords.words("spanish"))
    STOP_EXTRA = {"spotify", "app", "aplicación", "funciona", "función"}
    return STOP_ES.union(STOP_EXTRA)

STOPWORDS = cargar_stopwords()

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
def generar_graficas_regresion():
    """
    Dashboard de Regresión Lineal:
      - scatter + recta de regresión
      - histograma de residuos
      - real vs predicción
    """
    try:
        df = df_spotify()
    except Exception:
        return None, None, "No se pudo leer el dataset de Spotify."

    if COLUMNA_FEATURE_X not in df.columns or COLUMNA_STREAMS not in df.columns:
        cols = ", ".join(df.columns)
        return None, None, f"Revisa nombres de columnas. Columnas actuales: {cols}"

    X = df[[COLUMNA_FEATURE_X]]
    y = df[COLUMNA_STREAMS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    imagenes = {}

    # 1) Scatter + recta
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.scatter(X_test, y_test, alpha=0.5, label="Real")
    ax1.plot(X_test, y_pred, color="red", linewidth=2, label="Predicción")
    ax1.set_xlabel(COLUMNA_FEATURE_X)
    ax1.set_ylabel("streams")
    ax1.set_title("Streams vs " + COLUMNA_FEATURE_X)
    ax1.legend()
    imagenes["modelo"] = fig_to_base64(fig1)
    plt.close(fig1)

    # 2) Histograma de residuos
    residuos = y_test - y_pred
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.hist(residuos, bins=30, color="#3498db", alpha=0.8)
    ax2.set_title("Distribución de residuos")
    ax2.set_xlabel("Residuo (real - predicho)")
    ax2.set_ylabel("Frecuencia")
    imagenes["residuos"] = fig_to_base64(fig2)
    plt.close(fig2)

    # 3) Real vs predicho
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.scatter(y_test, y_pred, alpha=0.5)
    ax3.set_xlabel("Streams reales")
    ax3.set_ylabel("Streams predichos")
    ax3.set_title("Real vs Predicción")
    imagenes["real_vs_pred"] = fig_to_base64(fig3)
    plt.close(fig3)

    metricas = {
        "r2": round(r2, 4),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
    }

    return imagenes, metricas, None

# ------------------------ ÁRBOL DE DECISIÓN ------------------------ #
def generar_graficas_arbol():
    """
    Dashboard de Árbol de Decisión:
      - árbol completo
      - importancia de variables
    """
    try:
        df = df_spotify()
    except Exception:
        return None, None, "No se pudo leer el dataset de Spotify."

    for col in [COLUMNA_FEATURE_X, COLUMNA_FEATURE_X2, COLUMNA_STREAMS]:
        if col not in df.columns:
            cols = ", ".join(df.columns)
            return None, None, f"Falta la columna {col}. Columnas actuales: {cols}"

    X = df[[COLUMNA_FEATURE_X, COLUMNA_FEATURE_X2]]
    y = df[COLUMNA_STREAMS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    arbol = DecisionTreeRegressor(max_depth=4, random_state=42)
    arbol.fit(X_train, y_train)

    score = arbol.score(X_test, y_test)

    imagenes = {}

    # 1) Árbol
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    plot_tree(
        arbol,
        feature_names=[COLUMNA_FEATURE_X, COLUMNA_FEATURE_X2],
        filled=True,
        rounded=True,
        fontsize=6,
        ax=ax1,
    )
    ax1.set_title("Árbol de decisión - Predicción de streams")
    imagenes["arbol"] = fig_to_base64(fig1)
    plt.close(fig1)

    # 2) Importancia de variables
    importancias = arbol.feature_importances_
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar([COLUMNA_FEATURE_X, COLUMNA_FEATURE_X2], importancias)
    ax2.set_title("Importancia de variables")
    ax2.set_ylabel("Importancia")
    imagenes["importancias"] = fig_to_base64(fig2)
    plt.close(fig2)

    metricas = {
        "r2": round(score, 4),
        "profundidad": arbol.get_depth(),
        "n_nodos": arbol.tree_.node_count,
    }

    return imagenes, metricas, None


# ----------------------------- K-MEANS ------------------------------ #
def generar_graficas_kmeans():
    """
    Dashboard de K-means:
      - scatter de clusters
      - tamaño de cada cluster
    """
    try:
        df = df_spotify()
    except Exception:
        return None, None, "No se pudo leer el dataset de Spotify."

    for col in [COLUMNA_FEATURE_X, COLUMNA_FEATURE_X2]:
        if col not in df.columns:
            cols = ", ".join(df.columns)
            return None, None, f"Falta la columna {col}. Columnas actuales: {cols}"

    X = df[[COLUMNA_FEATURE_X, COLUMNA_FEATURE_X2]]

    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    imagenes = {}

    # 1) Scatter de clusters
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    scatter = ax1.scatter(
        X[COLUMNA_FEATURE_X], X[COLUMNA_FEATURE_X2], c=clusters, alpha=0.6
    )
    centers = kmeans.cluster_centers_
    ax1.scatter(centers[:, 0], centers[:, 1], s=200, marker="X")
    ax1.set_xlabel(COLUMNA_FEATURE_X)
    ax1.set_ylabel(COLUMNA_FEATURE_X2)
    ax1.set_title("Clusters K-means")
    imagenes["clusters"] = fig_to_base64(fig1)
    plt.close(fig1)

    # 2) Tamaño de clusters
    import numpy as np
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

    return imagenes, metricas, None


# ------------------------ ANÁLISIS DE SENTIMIENTO ------------------- #
def generar_graficas_sentimiento():
    """
    Genera TODAS las gráficas del módulo de sentimiento:
    - score
    - longitud comentarios
    - top palabras
    - wordcloud

    Devuelve:
      {
        "score": base64_img,
        "longitud": base64_img,
        "top_palabras": base64_img,
        "wordcloud": base64_img
      }
    """
    try:
        df = pd.read_csv(RUTA_DATA_REVIEWS)
    except:
        return None, "No se pudo leer el archivo de reseñas"

    if "content" not in df.columns or "score" not in df.columns:
        return None, "Faltan columnas necesarias ('content' y 'score')"

    imagenes = {}

    # ============ 1) DISTRIBUCIÓN DE SCORE ============
    conteo_score = df["score"].value_counts().sort_index()

    fig1, ax1 = plt.subplots(figsize=(6,4))
    conteo_score.plot(kind="bar", ax=ax1)
    ax1.set_title("Distribución de score")
    ax1.set_xlabel("Score")
    ax1.set_ylabel("Número de reseñas")
    imagenes["score"] = fig_to_base64(fig1)
    plt.close(fig1)

    # ============ 2) LONGITUD DE COMENTARIOS ============
    longitudes = df["content"].dropna().astype(str).apply(lambda x: len(x.split()))

    fig2, ax2 = plt.subplots(figsize=(6,4))
    ax2.hist(longitudes, bins=30, color="#3498db", alpha=0.8)
    ax2.set_title("Distribución de longitud de comentarios")
    ax2.set_xlabel("Número de palabras")
    ax2.set_ylabel("Cantidad de reseñas")
    imagenes["longitud"] = fig_to_base64(fig2)
    plt.close(fig2)

    # ============ 3) TOP PALABRAS (sin stopwords) ============
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

        fig3, ax3 = plt.subplots(figsize=(6,4))
        ax3.barh(palabras_top[::-1], frecs[::-1])
        ax3.set_title("Top 15 palabras más mencionadas")
        ax3.set_xlabel("Frecuencia")
        imagenes["top_palabras"] = fig_to_base64(fig3)
        plt.close(fig3)
    else:
        imagenes["top_palabras"] = None

    # ============ 4) WORDCLOUD ============
    texto_total = " ".join(textos).lower()
    texto_total = re.sub(r"[^a-záéíóúñü ]", " ", texto_total)

    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        colormap="viridis",
        max_words=200
    ).generate(texto_total)

    fig4 = plt.figure(figsize=(12,5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    imagenes["wordcloud"] = fig_to_base64(fig4)
    plt.close(fig4)

    return imagenes, None
# --------------------------------------------------------------------
# RUTAS FLASK
# --------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/regresion")
def vista_regresion():
    imagenes, metricas, error = generar_graficas_regresion()
    return render_template("regresion.html", imagenes=imagenes, metricas=metricas, error=error)

@app.route("/arbol")
def vista_arbol():
    imagenes, metricas, error = generar_graficas_arbol()
    return render_template("arbol.html", imagenes=imagenes, metricas=metricas, error=error)


@app.route("/kmeans")
def vista_kmeans():
    imagenes, metricas, error = generar_graficas_kmeans()
    return render_template("kmeans.html", imagenes=imagenes, metricas=metricas, error=error)

@app.route("/sentimiento")
def vista_sentimiento():
    imagenes, error = generar_graficas_sentimiento()

    return render_template(
        "sentimiento.html",
        imagenes=imagenes,
        error=error
    )

@app.route("/top_palabras")
def vista_top_palabras():
    grafico, error = grafico_top_palabras()
    return render_template(
        "top_palabras.html",
        grafico=grafico,
        error=error
    )


@app.route("/wordcloud")
def vista_wordcloud():
    grafico, error = grafico_wordcloud()
    return render_template(
        "wordcloud.html",
        grafico=grafico,
        error=error
    )


@app.route("/longitud")
def vista_longitud():
    grafico, error = grafico_longitud_comentarios()
    return render_template(
        "longitud.html",
        grafico=grafico,
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