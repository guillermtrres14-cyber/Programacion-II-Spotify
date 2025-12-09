from sklearn.cluster import KMeans
import numpy as np


def generar_graficas_kmeans():
    df_clean, numeric_cols, feature_cols = cargar_spotify_limpio()

    # Usamos dos dimensiones para poder graficar:
    x1 = "Streams per Listener"
    x2 = "Recent Streams Ratio"

    X = df_clean[[x1, x2]]

    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    imagenes = {}

    # Scatter de clusters
    fig_a, ax_a = plt.subplots(figsize=(6, 4))
    ax_a.scatter(X[x1], X[x2], c=clusters, alpha=0.6)
    centers = kmeans.cluster_centers_
    ax_a.scatter(centers[:, 0], centers[:, 1], s=200, marker="X")
    ax_a.set_xlabel(x1)
    ax_a.set_ylabel(x2)
    ax_a.set_title("Clusters K-means")
    imagenes["clusters"] = fig_to_base64(fig_a)
    plt.close(fig_a)

    # Tamaño de clusters
    sizes = np.bincount(clusters, minlength=k)
    fig_b, ax_b = plt.subplots(figsize=(6, 4))
    ax_b.bar(range(k), sizes)
    ax_b.set_xlabel("Cluster")
    ax_b.set_ylabel("Número de elementos")
    ax_b.set_title("Tamaño de cada cluster")
    imagenes["tamanos"] = fig_to_base64(fig_b)
    plt.close(fig_b)

    metricas = {
        "k": k,
        "inercia": round(kmeans.inertia_, 2),
    }

    return imagenes, metricas, None