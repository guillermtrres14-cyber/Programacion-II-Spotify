import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def run_kmeans(data_path: str) -> dict:
    df = pd.read_csv(data_path, encoding="latin-1")

    # Columnas numéricas útiles para clusters
    numeric_cols = [
        "Release Year",
        "Monthly Listeners (Millions)",
        "Total Streams (Millions)",
        "Total Hours Streamed (Millions)",
        "Avg Stream Duration (Min)",
        "Streams Last 30 Days (Millions)",
        "Skip Rate (%)"
    ]

    # Verificar que existan las columnas
    for col in numeric_cols:
        if col not in df.columns:
            raise ValueError(f"Falta la columna numérica: {col}")

    X = df[numeric_cols]

    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Entrenamiento de K-means
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    labels = kmeans.labels_

    # Tamaño de cada cluster
    tamaños = {int(i): int(sum(labels == i)) for i in range(k)}

    # Ejemplos de los primeros 10
    ejemplos = []
    for i in range(min(10, len(df))):
        ejemplos.append(
            f"{df.loc[i, 'Country']} | {df.loc[i, 'Artist']} | Cluster {labels[i]}"
        )

    return {
        "tipo": "K-means – Spotify",
        "columnas": numeric_cols,
        "n_clusters": k,
        "inercia": round(float(kmeans.inertia_), 3),
        "tamaño_clusters": tamaños,
        "ejemplos": ejemplos
    }