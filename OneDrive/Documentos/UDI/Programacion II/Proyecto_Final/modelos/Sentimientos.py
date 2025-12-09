import os
import re
from collections import Counter

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Para generar imágenes sin abrir ventanas
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob

# Rutas base
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "spotify_reviews.csv")
STATIC_IMG = os.path.join(BASE_DIR, "static", "img")
os.makedirs(STATIC_IMG, exist_ok=True)


def analizar_sentimientos(texto):
    """Clasifica el sentimiento en Positivo, Negativo o Neutral usando TextBlob."""
    polarity = TextBlob(str(texto)).sentiment.polarity
    if polarity > 0.1:
        return "Positivo"
    elif polarity < -0.1:
        return "Negativo"
    return "Neutral"


# STOPWORDS en inglés y español ampliadas
STOPWORDS = {
    # inglés
    "the","and","to","a","of","is","it","in","for","an","with","on","at","from",
    "that","this","was","were","be","been","are","am","as","so","but","if","or",
    "my","you","your","our","their","they","them","we","me","i","its",
    "can","could","would","should","very","really","just","also",
    # español
    "el","la","los","las","un","una","unos","unas","y","o","a","de","del","al",
    "que","se","es","en","por","para","con","sin","me","mi","tu","su","sus",
    "lo","como","más","menos","esto","eso","esta","este","estos","estas",
    # otras genéricas
    "app","spotify","application","review","reviews","user","users",
    "etc"
}


def limpiar_y_tokenizar(texto: str):
    """Limpia el texto y devuelve una lista de palabras útiles (sin stopwords, mín. 4 letras)."""
    texto = texto.lower()
    # Dejar solo letras y espacios
    texto = re.sub(r"[^a-zA-Záéíóúñ ]", " ", texto)
    palabras = texto.split()

    palabras_limpias = [
        p for p in palabras
        if p not in STOPWORDS
        and len(p) >= 4    # al menos 4 letras
        and not p.isnumeric()
    ]
    return palabras_limpias


def run_sentimientos():
    """
    Lee spotify_reviews.csv, calcula sentimiento, genera gráficas y devuelve
    las rutas de las imágenes + datos clave.
    """
    try:
        df = pd.read_csv(DATA_PATH)

        if "content" not in df.columns:
            raise ValueError("El CSV no tiene una columna llamada 'content'.")

        # 1) Clasificar sentimiento
        df["sentimiento"] = df["content"].apply(analizar_sentimiento)

        # Conteo de cada tipo de sentimiento
        conteo = df["sentimiento"].value_counts()

        # ================== GRÁFICA 1: PIE CHART ==================
        plt.figure(figsize=(7, 6))
        plt.pie(
            conteo,
            labels=conteo.index,
            autopct="%1.1f%%",
            startangle=140,
            colors=["#2ecc71", "#e74c3c", "#f1c40f"]  # verde, rojo, amarillo
        )
        plt.title("Distribución del Sentimiento en Reseñas de Spotify")
        pie_path = os.path.join(STATIC_IMG, "sent_pie.png")
        plt.savefig(pie_path, dpi=130)
        plt.close()

        # ================== GRÁFICA 2: SCORE PROMEDIO ==================
        # Asegurar que 'score' sea numérico
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        mean_scores = df.groupby("sentimiento")["score"].mean()

        plt.figure(figsize=(7, 5))
        mean_scores.plot(kind="bar", color=["#2ecc71", "#f1c40f", "#e74c3c"])
        plt.title("Score promedio por tipo de sentimiento")
        plt.ylabel("Score")
        plt.xticks(rotation=0)
        bar_path = os.path.join(STATIC_IMG, "sent_bar.png")
        plt.savefig(bar_path, dpi=130, bbox_inches="tight")
        plt.close()

        # ================== GRÁFICA 3: WORDCLOUD ==================
        texto_total = " ".join(df["content"].astype(str).tolist())
        wc = WordCloud(
            background_color="white",
            colormap="viridis",
            width=1000,
            height=600
        ).generate(texto_total)

        plt.figure(figsize=(10, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        wc_path = os.path.join(STATIC_IMG, "sent_wc.png")
        plt.savefig(wc_path, dpi=130, bbox_inches="tight")
        plt.close()

        # ================== GRÁFICA 4: TOP PALABRAS ÚTILES ==================
        todas_limpias = []
        for texto in df["content"]:
            todas_limpias.extend(limpiar_y_tokenizar(str(texto)))

        top_20 = Counter(todas_limpias).most_common(20)
        palabras_x = [p for p, _ in top_20]
        valores_y = [c for _, c in top_20]

        plt.figure(figsize=(10, 6))
        plt.barh(palabras_x, valores_y, color="#3498db")
        plt.gca().invert_yaxis()
        plt.title("Top 20 palabras más mencionadas (filtradas)")
        plt.xlabel("Frecuencia")
        top_words_path = os.path.join(STATIC_IMG, "sent_top_words.png")
        plt.savefig(top_words_path, dpi=130, bbox_inches="tight")
        plt.close()

        return {
            "ok": True,
            "conteo": conteo.to_dict(),
            "top_words": top_20,
            "images": {
                "pie": "img/sent_pie.png",
                "bar": "img/sent_bar.png",
                "wc": "img/sent_wc.png",
                "top_words": "img/sent_top_words.png"
            }
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "conteo": {},
            "top_words": [],
            "images": {}
        }