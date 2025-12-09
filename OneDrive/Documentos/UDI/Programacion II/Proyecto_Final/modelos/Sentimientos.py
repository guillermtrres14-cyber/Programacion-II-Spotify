def run_sentimiento(csv_path):
    import pandas as pd
    from textblob import TextBlob
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    import nltk
    nltk.download("punkt_tab")
    nltk.download("movie_reviews")

    df = pd.read_csv(csv_path)

    # Preprocesar comentarios
    df["clean"] = df["review"].astype(str)

    # Sentiment polarity
    df["polarity"] = df["clean"].apply(lambda x: TextBlob(x).sentiment.polarity)

    # Clasificación
    df["sentiment"] = df["polarity"].apply(
        lambda p: "positivo" if p > 0 else ("negativo" if p < 0 else "neutral")
    )

    # ======== GRÁFICAS ========
    # PIE CHART
    sentiment_counts = df["sentiment"].value_counts()
    plt.figure(figsize=(6, 6))
    sentiment_counts.plot(kind="pie", autopct="%1.1f%%")
    plt.title("Distribución de Sentimientos")
    plt.ylabel("")
    plt.savefig("static/images/sentiment_pie.png")
    plt.close()

    # BAR CHART
    plt.figure(figsize=(6, 4))
    sentiment_counts.plot(kind="bar", color=["green", "red", "gray"])
    plt.title("Sentimientos Totales")
    plt.savefig("static/images/sentiment_bar.png")
    plt.close()

    # WORD CLOUD
    text = " ".join(df["clean"])
    wc = WordCloud(background_color="black", width=800, height=400).generate(text)
    wc.to_file("static/images/sentiment_wordcloud.png")

    # TOP WORDS
    words = text.split()
    freq = pd.Series(words).value_counts().head(20)

    plt.figure(figsize=(8, 6))
    freq.plot(kind="bar")
    plt.title("Palabras Más Usadas")
    plt.savefig("static/images/sentiment_top_words.png")
    plt.close()

    return {
        "ok": True,
        "images": {
            "pie": "images/sentiment_pie.png",
            "bar": "images/sentiment_bar.png",
            "wc": "images/sentiment_wordcloud.png",
            "top_words": "images/sentiment_top_words.png",
        }
    }