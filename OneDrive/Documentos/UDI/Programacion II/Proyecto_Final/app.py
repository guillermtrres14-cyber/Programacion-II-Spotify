from flask import Flask, render_template, url_for

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/regresion")
def regresion():
    img_path = url_for('static', filename='img/regresion_streams.png')
    return render_template("regresion.html", img_regresion=img_path)

@app.route("/arbol")
def arbol():
    img_path = url_for('static', filename='img/arbol_decision.png')
    return render_template("arbol.html", img_arbol=img_path)

@app.route("/kmeans")
def kmeans():
    img_path = url_for('static', filename='img/kmeans_clusters.png')
    return render_template("kmeans.html", img_kmeans=img_path)

@app.route("/sentimiento")
def sentimiento():
    img_path = url_for('static', filename='img/sentimiento_reviews.png')
    return render_template("sentimiento.html", img_sentimiento=img_path)


if __name__ == "__main__":
    app.run(debug=True)