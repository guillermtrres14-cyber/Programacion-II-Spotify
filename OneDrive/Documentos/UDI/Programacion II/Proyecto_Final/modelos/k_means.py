import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Rutas base del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_IMG_DIR = os.path.join(BASE_DIR, "static", "img")
os.makedirs(STATIC_IMG_DIR, exist_ok=True)


def run_kmeans(data_path: str | None = None) -> dict:
    try:
        # 1. Ruta del dataset
        if data_path is None:
            data_path = os.path.join(DATA_DIR, "Spotify_2024_Global_Streaming_Data.csv")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"No se encontró el dataset en: {data_path}")

        df = pd.read_csv(data_path, encoding="latin-1")

        # 2. Columnas numéricas para clustering
        numeric_cols = [
            "Release Year",
            "Monthly Listeners (Millions)",
            "Total Streams (Millions)",
            "Total Hours Streamed (Millions)",
            "Avg Stream Duration (Min)",
            "Streams Last 30 Days (Millions)",
            "Skip Rate (%)"
        ]

        for col in numeric_cols:
            if col not in df.columns:
                raise ValueError(f"Falta la columna numérica: {col}")

        X = df[numeric_cols].dropna()

        # 3. Escalado
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 4. Entrenamiento K-means
        k = 3
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)

        labels = kmeans.labels_
        inertia = float(kmeans.inertia_)

        # Tamaño de cada cluster
        tamaño_clusters = {int(i): int(np.sum(labels == i)) for i in range(k)}

        # 5. Ejemplos (usando solo filas válidas después del dropna)
        X_valid_indices = X.index
        ejemplos = []
        for idx, row_idx in enumerate(X_valid_indices[:10]):
            country = df.loc[row_idx, "Country"] if "Country" in df.columns else "N/A"
            artist = df.loc[row_idx, "Artist"] if "Artist" in df.columns else "N/A"
            ejemplos.append(
                f"{country} | {artist} | Cluster {int(labels[idx])}"
            )

        # 6. PCA para visualización 2D
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        # ==== Gráfico 1: Dispersión PCA coloreada por cluster ====
        plt.figure(figsize=(7, 5))
        for cluster_id in range(k):
            mask = labels == cluster_id
            plt.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                alpha=0.6,
                label=f"Cluster {cluster_id}"
            )
        plt.xlabel("Componente principal 1")
        plt.ylabel("Componente principal 2")
        plt.title("K-means – Clusters en espacio PCA")
        plt.legend()
        plt.tight_layout()
        clusters_path = os.path.join(STATIC_IMG_DIR, "kmeans_clusters_pca.png")
        plt.savefig(clusters_path, dpi=120)
        plt.close()

        # ==== Gráfico 2: Tamaño de clusters ====
        plt.figure(figsize=(6, 4))
        plt.bar(
            list(tamaño_clusters.keys()),
            list(tamaño_clusters.values())
        )
        plt.xlabel("Cluster")
        plt.ylabel("Número de observaciones")
        plt.title("K-means – Tamaño de cada cluster")
        plt.tight_layout()
        sizes_path = os.path.join(STATIC_IMG_DIR, "kmeans_tamaño_clusters.png")
        plt.savefig(sizes_path, dpi=120)
        plt.close()

        return {
            "ok": True,
            "tipo": "K-means – Spotify",
            "columnas": numeric_cols,
            "n_clusters": k,
            "inercia": round(inertia, 3),
            "tamaño_clusters": tamaño_clusters,
            "ejemplos": ejemplos,
            "images": {
                "clusters_pca": "img/kmeans_clusters_pca.png",
                "tamaño_clusters": "img/kmeans_tamaño_clusters.png"
            }
        }

    except Exception as e:
        return {
            "ok": False,
            "tipo": "K-means – Spotify",
            "error": f"Error al ejecutar K-means: {str(e)}",
            "columnas": [],
            "n_clusters": 0,
            "inercia": None,
            "tamaño_clusters": {},
            "ejemplos": [],
            "images": {}
        }