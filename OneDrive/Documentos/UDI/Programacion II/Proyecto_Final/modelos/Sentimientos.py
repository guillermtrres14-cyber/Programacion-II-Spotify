import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
import nltk

# Descargar recursos necesarios de NLTK (silencioso para no llenar la consola)
nltk.download("punkt", quiet=True)
nltk.download("movie_reviews", quiet=True)


def run_sentimiento(csv_path: str) -> dict:
    """
    Lee un CSV con reseñas de usuarios y genera:
    - Distribución de sentimientos (pie + bar)
    - Nube de palabras
    - Top de palabras más usadas

    Intenta detectar automáticamente la columna de texto:
    'review', 'Review', 'text', 'Text', 'comment', 'Comentario', etc.
    """

    try:
        # Lee el CSV (ajusta encoding si fuera necesario)
        df = pd.read_csv(csv_path, encoding="latin-1")

        # 1) Detectar la columna de texto
        candidate_cols = [
            "review",
            "Review",
            "review_text",
            "text",
            "Text",
            "comment",
            "Comment",
            "Comentario",
        ]

        review_col = None
        for c in candidate_cols:
            if c in df.columns:
                review_col = c
                break

        # Si no encuentra ninguna de esas, usa la primera columna de tipo texto
        if review_col is None:
            text_cols = df.select_dtypes(include="object").columns
            if len(text_cols) == 0:
                return {
                    "ok": False,
                    "error": "No se encontró ninguna columna de texto en el CSV."
                }
            review_col = text_cols[0]

        # Limpieza básica
        df["clean"] = df[review_col].astype(str)

        # 2) Cálculo de polaridad con TextBlob
        df["polarity"] = df["clean"].apply(lambda x: TextBlob(x).sentiment.polarity)

        def etiqueta(p):
            if p > 0.05:
                return "positivo"
            if p < -0.05:
                return "negativo"
            return "neutral"

        df["sentiment"] = df["polarity"].apply(etiqueta)

        # Conteo ordenado
        sentiment_counts = df["sentiment"].value_counts().reindex(
            ["positivo", "neutral", "negativo"], fill_value=0
        )

        # Carpeta de salida de imágenes
        img_dir = os.path.join("static", "images")
        os.makedirs(img_dir, exist_ok=True)

        # ---------- Gráfico PIE ----------
        plt.figure(figsize=(5, 5))
        sentiment_counts.plot(kind="pie", autopct="%1.1f%%")
        plt.title("Distribución de Sentimientos")
        plt.ylabel("")
        pie_path = os.path.join(img_dir, "sentiment_pie.png")
        plt.savefig(pie_path, bbox_inches="tight", transparent=True)
        plt.close()

        # ---------- Gráfico BARRAS ----------
        plt.figure(figsize=(6, 4))
        sentiment_counts.plot(kind="bar")
        plt.title("Sentimientos Totales")
        plt.xticks(rotation=0)
        bar_path = os.path.join(img_dir, "sentiment_bar.png")
        plt.savefig(bar_path, bbox_inches="tight", transparent=True)
        plt.close()

        # ---------- WordCloud ----------
        texto_total = " ".join(df["clean"])
        wc = WordCloud(
            background_color="black",
            width=1000,
            height=500,
            max_words=150,
        ).generate(texto_total)
        wc_path = os.path.join(img_dir, "sentiment_wordcloud.png")
        wc.to_file(wc_path)

        # ---------- Top palabras (filtrando palabras inútiles) ----------
        palabras = pd.Series(texto_total.split())
        stop = set([
            "the", "and", "to", "a", "of", "in", "is", "for", "on",
            "la", "el", "y", "que", "de", "en", "un", "una", "es",
            "por", "con"
        ])
        palabras_filtradas = palabras[~palabras.str.lower().isin(stop)]
        freq = palabras_filtradas.value_counts().head(15)

        plt.figure(figsize=(8, 4))
        freq.sort_values().plot(kind="barh")
        plt.title("Palabras más usadas")
        plt.tight_layout()
        top_path = os.path.join(img_dir, "sentiment_top_words.png")
        plt.savefig(top_path, transparent=True)
        plt.close()

        # Devolvemos rutas relativas para usarlas con url_for('static', ...)
        return {
            "ok": True,
            "review_col": review_col,
            "n_rows": int(len(df)),
            "resumen": sentiment_counts.to_dict(),
            "images": {
                "pie": "images/sentiment_pie.png",
                "bar": "images/sentiment_bar.png",
                "wc": "images/sentiment_wordcloud.png",
                "top_words": "images/sentiment_top_words.png",
            },
        }

    except Exception as e:
        return {
            "ok": False,
            "error": f"Error al procesar el CSV: {e}",
        }