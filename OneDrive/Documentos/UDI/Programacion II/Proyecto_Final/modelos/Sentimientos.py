import os
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud


def _detectar_columna_texto(df: pd.DataFrame) -> str:
    """
    Intenta encontrar la columna de texto:
    - 'review', 'text' o 'comment'
    - si no, la primera columna de tipo object (string).
    """
    for col in ["review", "text", "comment"]:
        if col in df.columns:
            return col

    obj_cols = df.select_dtypes(include="object").columns
    if len(obj_cols) == 0:
        raise ValueError(
            "El CSV no tiene ninguna columna de texto (por ejemplo 'review')."
        )
    return obj_cols[0]


def _clasificar_polaridad(p: float) -> str:
    """Convierte polaridad numérica en categoría de sentimiento."""
    if p > 0.05:
        return "Positivo"
    elif p < -0.05:
        return "Negativo"
    else:
        return "Neutral"


def run_sentimiento(csv_path: str):
    """
    Ejecuta el análisis de sentimiento y genera las gráficas necesarias.
    Devuelve un diccionario con las rutas (relativas a /static) de las imágenes.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {"ok": False, "error": f"No se pudo leer el CSV: {e}"}

    try:
        text_col = _detectar_columna_texto(df)
    except ValueError as e:
        # Mensaje que veías en pantalla
        return {"ok": False, "error": str(e)}

    # --- Preprocesado de texto ------------------------------------------------
    df[text_col] = df[text_col].astype(str).fillna("")

    # Polaridad con TextBlob
    df["polarity"] = df[text_col].apply(lambda t: TextBlob(t).sentiment.polarity)

    # Categorías de sentimiento
    df["sentiment"] = df["polarity"].apply(_clasificar_polaridad)

    # Conteos y promedio
    order = ["Positivo", "Neutral", "Negativo"]
    sentiment_counts = (
        df["sentiment"].value_counts().reindex(order).fillna(0).astype(int)
    )
    mean_score = (
        df.groupby("sentiment")["polarity"].mean().reindex(order).fillna(0.0)
    )

    # Carpeta de salida
    img_dir = os.path.join("static", "images")
    os.makedirs(img_dir, exist_ok=True)

    # ==== 1) PIE: Distribución de sentimientos ================================
    plt.style.use("dark_background")
    colors = ["#1ED760", "#FFFFFF", "#FF4B4B"]  # verde, blanco, rojo

    plt.figure(figsize=(5, 5))
    sentiment_counts.plot(
        kind="pie",
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        textprops={"fontsize": 11},
    )
    plt.title("Distribución de sentimientos")
    plt.ylabel("")
    pie_path = os.path.join(img_dir, "sentiment_pie.png")
    plt.tight_layout()
    plt.savefig(pie_path, dpi=120)
    plt.close()

    # ==== 2) BARRAS: Score promedio por sentimiento ===========================
    plt.figure(figsize=(6, 4))
    bars = mean_score.plot(kind="bar", color=colors)
    plt.title("Score promedio por sentimiento (polarity)")
    plt.xlabel("Sentimiento")
    plt.ylabel("Polarity promedio")

    # Etiquetas numéricas encima de cada barra
    for patch, value in zip(bars.patches, mean_score.values):
        bars.annotate(
            f"{value:.2f}",
            (patch.get_x() + patch.get_width() / 2.0, value),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    bar_score_path = os.path.join(img_dir, "sentiment_score.png")
    plt.tight_layout()
    plt.savefig(bar_score_path, dpi=120)
    plt.close()

    # ==== 3) Nube de palabras ================================================
    text = " ".join(df[text_col])
    wc = WordCloud(
        width=800,
        height=400,
        background_color="black",
        colormap="viridis"
    ).generate(text)
    wc_path = os.path.join(img_dir, "sentiment_wordcloud.png")
    wc.to_file(wc_path)

    # ==== 4) Top palabras =====================================================
    words = [w.lower() for w in text.split() if len(w) > 3]
    freq = Counter(words)
    top_words = freq.most_common(10)

    labels, values = zip(*top_words) if top_words else ([], [])

    plt.figure(figsize=(8, 4))
    plt.barh(range(len(values)), values)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Frecuencia")
    plt.title("Palabras más mencionadas")
    plt.tight_layout()
    top_words_path = os.path.join(img_dir, "sentiment_top_words.png")
    plt.savefig(top_words_path, dpi=120)
    plt.close()

    # Diccionario de salida compatible con index.html
    return {
        "ok": True,
        "images": {
            "pie": "images/" + os.path.basename(pie_path),
            "bar": "images/" + os.path.basename(bar_score_path),
            "wc": "images/" + os.path.basename(wc_path),
            "top_words": "images/" + os.path.basename(top_words_path),
        },
        "summary": {
            "total_reviews": int(len(df)),
            "sentiment_counts": sentiment_counts.to_dict(),
            "mean_score": {k: float(v) for k, v in mean_score.items()},
        },
    }