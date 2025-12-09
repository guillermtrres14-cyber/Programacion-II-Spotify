from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def generar_graficas_regresion():
    df_clean, numeric_cols, feature_cols = cargar_spotify_limpio()

    # ======== 1) Modelo multivariable como en el notebook ========
    X = df_clean[feature_cols]
    y = df_clean[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred = lin_reg.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    imagenes = {}

    # --- Gráfico A: Oyentes mensuales vs streams totales ---
    fig_a, ax_a = plt.subplots(figsize=(6, 4))
    ax_a.scatter(
        df_clean["Monthly Listeners (Millions)"],
        df_clean["Total Streams (Millions)"],
        alpha=0.5,
    )
    ax_a.set_xlabel("Monthly Listeners (Millions)")
    ax_a.set_ylabel("Total Streams (Millions)")
    ax_a.set_title("Oyentes mensuales vs Streams totales")
    imagenes["scatter_listeners"] = fig_to_base64(fig_a)
    plt.close(fig_a)

    # --- Gráfico B: Streams per Listener vs streams totales ---
    fig_b, ax_b = plt.subplots(figsize=(6, 4))
    ax_b.scatter(
        df_clean["Streams per Listener"],
        df_clean["Total Streams (Millions)"],
        alpha=0.5,
    )
    ax_b.set_xlabel("Streams per Listener")
    ax_b.set_ylabel("Total Streams (Millions)")
    ax_b.set_title("Intensidad de escucha vs Streams totales")
    imagenes["scatter_intensidad"] = fig_to_base64(fig_b)
    plt.close(fig_b)

    # --- Gráfico C: Real vs Predicho con línea ideal ---
    fig_c, ax_c = plt.subplots(figsize=(6, 4))
    ax_c.scatter(y_test, y_pred, alpha=0.5, label="Predicciones")
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax_c.plot([min_val, max_val], [min_val, max_val], linestyle="--", label="Línea ideal")
    ax_c.set_xlabel(f"Valor real ({TARGET_COL})")
    ax_c.set_ylabel(f"Predicción ({TARGET_COL})")
    ax_c.set_title("Real vs Predicho – Regresión Lineal")
    ax_c.legend()
    imagenes["real_vs_pred"] = fig_to_base64(fig_c)
    plt.close(fig_c)

    # --- Gráfico D: Distribución de residuos ---
    residuals = y_test - y_pred
    fig_d, ax_d = plt.subplots(figsize=(6, 4))
    ax_d.hist(residuals, bins=30)
    ax_d.set_title("Distribución de residuos (error)")
    ax_d.set_xlabel("Error (real - predicho)")
    ax_d.set_ylabel("Frecuencia")
    imagenes["residuos"] = fig_to_base64(fig_d)
    plt.close(fig_d)

    # ======== 2) Regresión 1D con la mejor característica ========
    corr = df_clean[numeric_cols].corr()[TARGET_COL]
    corr_features = corr.drop(TARGET_COL).abs().sort_values(ascending=False)
    best_feature = corr_features.index[0]

    lin_1d = LinearRegression()
    X_best = df_clean[[best_feature]]
    y_all = df_clean[TARGET_COL]
    lin_1d.fit(X_best, y_all)

    x_min, x_max = X_best[best_feature].min(), X_best[best_feature].max()
    x_line = np.linspace(x_min, x_max, 100)
    X_line = pd.DataFrame({best_feature: x_line})
    y_line = lin_1d.predict(X_line)

    fig_e, ax_e = plt.subplots(figsize=(7, 5))
    ax_e.scatter(X_best[best_feature], y_all, alpha=0.3, label="Datos reales")
    ax_e.plot(x_line, y_line, linewidth=2, label="Línea de regresión")
    ax_e.set_xlabel(best_feature)
    ax_e.set_ylabel(TARGET_COL)
    ax_e.set_title(f"Línea de regresión usando {best_feature}")
    ax_e.legend()
    imagenes["linea_best"] = fig_to_base64(fig_e)
    plt.close(fig_e)

    metricas = {
        "mse": round(mse, 3),
        "mae": round(mae, 3),
        "r2": round(r2, 4),
        "mejor_feature": best_feature,
    }

    return imagenes, metricas, None