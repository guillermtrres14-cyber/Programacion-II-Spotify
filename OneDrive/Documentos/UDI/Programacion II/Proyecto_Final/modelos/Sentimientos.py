import os
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk

# --- Comprobar y descargar recursos de NLTK solo si faltan ---
def ensure_nltk_resource(path, download_name):
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(download_name)

# Tokenizador y corpus (no pasa nada si ya están descargados)
ensure_nltk_resource("tokenizers/punkt", "punkt")
ensure_nltk_resource("corpora/movie_reviews", "movie_reviews")

# Carpeta donde guardaremos las imágenes
IMAGES_DIR = os.path.join("static", "images")


def run_sentimiento(
    csv_path: str = os.path.join("data", "spotify_reviews.csv")
) -> dict:
    """
    Lee el CSV de reseñas, calcula polaridad con TextBlob,
    genera las gráficas y devuelve las rutas relativas de las imágenes.
    """

    # -------- Leer CSV --------
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except Exception as e:
        return {"ok": False, "error": f"No se pudo leer el CSV de reseñas: {e}"}

    if "review" not in df.columns:
        return {
            "ok": False,
            "error": "El CSV debe tener una columna llamada 'review'.",
        }

    # -------- Procesar texto --------
    df["clean"] = df["review"].astype(str).str.strip()

    # Polaridad con TextBlob
    df["polarity"] = df["clean"].apply(lambda x: TextBlob(x).sentiment.polarity)

    # Clasificación simple
    def label(p):
        if p > 0.05:
            return "positivo"
        elif p < -0.05:
            return "negativo"
        return "neutral"

    df["sentiment"] = df["polarity"].apply(label)

    # Asegurar carpeta de imágenes
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # -------- Gráfica 1: Pie de sentimientos --------
    sentiment_counts = df["sentiment"].value_counts()
    plt.figure(figsize=(5, 5))
    sentiment_counts.plot(kind="pie", autopct="%1.1f%%")
    plt.title("Distribución de Sentimientos")
    plt.ylabel("")
    pie_path_abs = os.path.join(IMAGES_DIR, "sentiment_pie.png")
    plt.tight_layout()
    plt.savefig(pie_path_abs, dpi=120, bbox_inches="tight")
    plt.close()

    # -------- Gráfica 2: Barras de totales --------
    plt.figure(figsize=(5, 4))
    sentiment_counts.plot(kind="bar", color=["#1ed760", "#ff4c4c", "#9ca3af"])
    plt.title("Sentimientos Totales")
    plt.xticks(rotation=0)
    bar_path_abs = os.path.join(IMAGES_DIR, "sentiment_bar.png")
    plt.tight_layout()
    plt.savefig(bar_path_abs, dpi=120, bbox_inches="tight")
    plt.close()

    # -------- Gráfica 3: Nube de palabras --------
    text = " ".join(df["clean"])

    wc = WordCloud(
        width=800,
        height=400,
        background_color="black",
        colormap="viridis",
    ).generate(text)

    wc_path_abs = os.path.join(IMAGES_DIR, "sentiment_wordcloud.png")
    wc.to_file(wc_path_abs)

    # -------- Gráfica 4: Top palabras útiles --------
    stopwords = {
        "the", "and", "to", "of", "for", "in", "is", "it", "on",
        "a", "an", "this", "that", "with",
        "la", "el", "los", "las", "de", "del", "y", "que", "con",
    }

    words = [
        w.lower()
        for w in text.split()
        if w.isalpha() and w.lower() not in stopwords
    ]

    if words:
        freq = pd.Series(words).value_counts().head(15)
        plt.figure(figsize=(6, 4))
        freq.plot(kind="bar")
        plt.title("Palabras más usadas")
        plt.xticks(rotation=45, ha="right")
        top_words_path_abs = os.path.join(IMAGES_DIR, "sentiment_top_words.png")
        plt.tight_layout()
        plt.savefig(top_words_path_abs, dpi=120, bbox_inches="tight")
        plt.close()
        top_words_rel = os.path.join("images", "sentiment_top_words.png")
    else:
        top_words_rel = None

    # Rutas RELATIVAS para usarlas en url_for('static', filename=...)
    return {
        "ok": True,
        "images": {
            "pie": os.path.join("images", "sentiment_pie.png"),
            "bar": os.path.join("images", "sentiment_bar.png"),
            "wc": os.path.join("images", "sentiment_wordcloud.png"),
            "top_words": top_words_rel,
        },
    }