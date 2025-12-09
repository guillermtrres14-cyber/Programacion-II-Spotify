import os
from flask import Flask, render_template

# === IMPORTS DE MODELOS ==============================

# Importo el módulo completo de regresión para poder adaptarme
# al nombre real de la función (run_regresion, run_regression_model, regresion, etc.)
import modelos.regresion as m_reg

# Estos tres ya los habíamos usado así:
from modelos.arbol import run_arbol
from modelos.k_means import run_kmeans
from modelos.sentimientos import run_sentimiento

# === CONFIGURACIÓN BÁSICA ============================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(BASE_DIR, "data", "Spotify_2024_Global_Streaming_Data.csv")
REVIEWS_FILE = os.path.join(BASE_DIR, "data", "spotify_reviews.csv")


# === HELPER: DETECTAR FUNCIÓN DE REGRESIÓN ===========

def ejecutar_regresion(data_path: str) -> dict:
    """
    Intenta llamar a la función correcta dentro de modelos/regresion.py
    sin importar cómo la hayas nombrado.
    """
    # PRIORIDAD: nombres que recomendamos
    posibles_nombres = [
        "run_regresion",
        "run_regression_model",
        "run_regresion_model",
        "regresion",
        "regression",
    ]

    for nombre in posibles_nombres:
        if hasattr(m_reg, nombre):
            fn = getattr(m_reg, nombre)
            return fn(data_path)

    # Si llegamos aquí, no hay ninguna función compatible
    raise RuntimeError(
        "No se encontró ninguna función de regresión válida en modelos/regresion.py. "
        "Define, por ejemplo, 'run_regresion(data_path)' y que retorne un dict con métricas e imágenes."
    )


# =====================================================
#                      RUTAS
# =====================================================

@app.route("/")
def index():
    """
    Pantalla principal: sólo muestra el dashboard general.
    No ejecuta modelos todavía (para que cargue rápido).
    """
    return render_template(
        "index.html",
        selected_model=None,
        metrics=None,
        results=None,
        sentimiento=None,
    )


# ---------------- REGRESIÓN LINEAL -------------------

@app.route("/regresion")
def vista_regresion():
    try:
        resultados = ejecutar_regresion(DATA_FILE)
        # Se asume que 'resultados' es un dict del tipo:
        # {
        #   "ok": True,
        #   "tipo": "Regresión Lineal ...",
        #   "R2": ...,
        #   "MSE": ...,
        #   "MAE": ...,
        #   "images": {
        #       "scatter": "images/reg_scatter.png",
        #       "importancia": "images/reg_importancia.png"
        #   }
        # }
        return render_template(
            "regresion.html",
            selected_model="Regresión Lineal",
            metrics=resultados,
            results=resultados,
        )
    except Exception as e:
        # Si algo falla, mostramos el error en la propia página de regresión
        return render_template(
            "regresion.html",
            selected_model="Regresión Lineal",
            metrics=None,
            results=None,
            error=str(e),
        )


# ---------------- ÁRBOL DE DECISIÓN ------------------

@app.route("/arbol")
def vista_arbol():
    try:
        resultados = run_arbol(DATA_FILE)
        return render_template(
            "arbol.html",
            selected_model="Árbol de Decisión",
            metrics=resultados,
            results=resultados,
        )
    except Exception as e:
        return render_template(
            "arbol.html",
            selected_model="Árbol de Decisión",
            metrics=None,
            results=None,
            error=str(e),
        )


# ------------------- K-MEANS -------------------------

@app.route("/kmeans")
def vista_kmeans():
    try:
        resultados = run_kmeans(DATA_FILE)
        return render_template(
            "kmeans.html",
            selected_model="K-means",
            metrics=resultados,
            results=resultados,
        )
    except Exception as e:
        return render_template(
            "kmeans.html",
            selected_model="K-means",
            metrics=None,
            results=None,
            error=str(e),
        )


# --------------- ANÁLISIS DE SENTIMIENTO --------------

@app.route("/sentimiento")
def vista_sentimiento():
    """
    Análisis de sentimiento sobre spotify_reviews.csv.
    """
    csv_path = REVIEWS_FILE

    try:
        sentimiento = run_sentimiento(csv_path)
        # Se asume que 'sentimiento' es un dict:
        # {
        #   "ok": True,
        #   "images": {
        #       "pie": "images/sentiment_pie.png",
        #       "bar": "images/sentiment_bar.png",
        #       "wc": "images/sentiment_wordcloud.png",
        #       "top_words": "images/sentiment_top_words.png"
        #   },
        #   ... (otros datos)
        # }
    except Exception as e:
        sentimiento = {
            "ok": False,
            "error": str(e),
            "images": {},
        }

    return render_template(
        "sentimiento.html",
        selected_model="Análisis de Sentimiento",
        sentimiento=sentimiento,
    )


# ------------------ DATASET --------------------------

@app.route("/dataset")
def vista_dataset():
    """
    Vista sencilla para ver una muestra del dataset principal.
    """
    import pandas as pd

    try:
        df = pd.read_csv(DATA_FILE, encoding="latin-1")
        sample = df.head(50).to_html(classes="table table-dark table-sm", index=False)
        return render_template("dataset.html", table_html=sample)
    except Exception as e:
        return render_template("dataset.html", table_html=None, error=str(e))


# =====================================================
#                   MAIN
# =====================================================

if __name__ == "__main__":
    # Sólo para entorno local de desarrollo
    app.run(debug=True)