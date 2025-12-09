import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud

# Carpeta donde se guardan las imágenes
IMG_DIR = Path("static/images")
IMG_DIR.mkdir(parents=True, exist_ok=True)


def _detectar_columna_texto(df: pd.DataFrame) -> str:
    """
    Devuelve el nombre de la columna que contiene el texto de la reseña.
    Acepta varios nombres comunes.
    """
    posibles = ["review", "content", "text", "comment", "comments"]
    for col in posibles:
        if col in df.columns:
            return col
    raise ValueError(
        "El CSV debe tener una columna de texto llamada "
        "'review', 'content' o similar."
    )


def run_sentimiento(csv_path: str) -> dict:
    try:
        # Leer CSV
        df = pd.read_csv(csv_path, encoding="utf-8")

        # === 1. Elegir la columna correcta de texto ===
        col_texto = _detectar_columna_texto(df)
        df["clean"] = df[col_texto].astype(str)

        # Si todo está vacío, mejor avisar
        if df["clean"].str.strip().eq("").all():
            return {
                "ok": False,
                "error": "No hay texto válido en la columna de reseñas.",
            }

        # === 2. Calcular polaridad y etiqueta de sentimiento ===
        df["polarity"] = df["clean"].apply(
            lambda x: TextBlob(x).sentiment.polarity
        )

        # Umbrales un poco más amplios para evitar que todo sea neutral
        def clasificar(p):
            if p > 0.1:
                return "positivo"
            elif p < -0.1:
                return "negativo"
            else:
                return "neutral"

        df["sentiment"] = df["polarity"].apply(clasificar)

        conteo = (
            df["sentiment"]
            .value_counts()
            .reindex(["positivo", "neutral", "negativo"], fill_value=0)
        )

        # === 3. Gráfico de pastel (distribución de sentimientos) ===
        plt.figure(figsize=(5, 5))
        colors = ["#1ED760", "#F1F1F1", "#FF4F4F"]  # verde, gris, rojo
        conteo.plot(
            kind="pie",
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
        )
        plt.title("Distribución de Sentimientos")
        plt.ylabel("")
        pie_path = IMG_DIR / "sentiment_pie.png"
        plt.savefig(pie_path, bbox_inches="tight", facecolor="black")
        plt.close()

        # === 4. Barra: score promedio por tipo de sentimiento ===
        mean_scores = (
            df.groupby("sentiment")["polarity"]
            .mean()
            .reindex(["positivo", "neutral", "negativo"])
        )

        plt.figure(figsize=(5, 4))
        mean_scores.plot(kind="bar")
        plt.title("Score promedio por sentimiento")
        plt.xticks(rotation=0)
        bar_path = IMG_DIR / "sentiment_bar.png"
        plt.savefig(bar_path, bbox_inches="tight", facecolor="black")
        plt.close()

        # === 5. Nube de palabras y palabras más usadas ===
        # Unimos todo el texto
        texto_total = " ".join(df["clean"])

        # Solo palabras alfabéticas (evitamos IDs tipo '4d89', números, etc.)
        tokens = re.findall(r"[A-Za-zÁÉÍÓÚáéíóúÑñü']{3,}", texto_total.lower())

        # Stopwords sencillas EN + ES para limpiar
        stopwords = {
            "the", "and", "for", "this", "that", "with", "you", "your",
            "are", "was", "but", "not", "very", "have", "has", "had",
            "los", "las", "unos", "unas", "muy", "pero", "para", "por",
            "con", "sin", "que", "como", "cuando", "donde", "sobre",
        }
        tokens = [t for t in tokens if t not in stopwords]

        # Nube de palabras
        if tokens:
            wc = WordCloud(
                width=900,
                height=400,
                background_color="black",
                colormap="viridis",
            ).generate(" ".join(tokens))
            wc_path = IMG_DIR / "sentiment_wordcloud.png"
            wc.to_file(wc_path)
        else:
            wc_path = IMG_DIR / "sentiment_wordcloud.png"
            # Nube vacía (por si acaso)
            WordCloud(width=900, height=400, background_color="black").to_file(
                wc_path
            )

        # Top 15 palabras más frecuentes
        freq = Counter(tokens)
        top = freq.most_common(15)
        palabras, frecuencias = zip(*top) if top else ([], [])

        plt.figure(figsize=(8, 4))
        plt.barh(range(len(palabras)), frecuencias)
        plt.yticks(range(len(palabras)), palabras)
        plt.gca().invert_yaxis()
        plt.title("Palabras más usadas")
        top_words_path = IMG_DIR / "sentiment_top_words.png"
        plt.savefig(top_words_path, bbox_inches="tight", facecolor="black")
        plt.close()

        return {
            "ok": True,
            "tipo": "Análisis de Sentimiento – Reseñas Spotify",
            "images": {
                # Lo que usarás en index.html: url_for('static', filename=...)
                "pie": "images/" + pie_path.name,
                "bar": "images/" + bar_path.name,
                "wc": "images/" + wc_path.name,
                "top_words": "images/" + top_words_path.name,
            },
            "resumen": {
                "total": int(len(df)),
                "positivos": int(conteo["positivo"]),
                "neutros": int(conteo["neutral"]),
                "negativos": int(conteo["negativo"]),
            },
        }

    except Exception as e:
        # Cualquier error se devuelve al template para mostrar el mensaje
        return {"ok": False, "error": str(e)}