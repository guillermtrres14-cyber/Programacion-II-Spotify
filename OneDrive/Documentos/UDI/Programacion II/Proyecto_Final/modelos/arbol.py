from sklearn.tree import DecisionTreeRegressor, plot_tree


def generar_graficas_arbol():
    df_clean, numeric_cols, feature_cols = cargar_spotify_limpio()

    X = df_clean[feature_cols]
    y = df_clean[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    tree_reg = DecisionTreeRegressor(max_depth=6, random_state=42)
    tree_reg.fit(X_train, y_train)

    y_pred_tree = tree_reg.predict(X_test)

    mse_tree = mean_squared_error(y_test, y_pred_tree)
    mae_tree = mean_absolute_error(y_test, y_pred_tree)
    r2_tree = r2_score(y_test, y_pred_tree)

    imagenes = {}

    # --- Gráfico A: Real vs Predicho – Árbol de decisión ---
    fig_a, ax_a = plt.subplots(figsize=(6, 5))
    ax_a.scatter(y_test, y_pred_tree, alpha=0.5, label="Predicciones")
    min_val = min(y_test.min(), y_pred_tree.min())
    max_val = max(y_test.max(), y_pred_tree.max())
    ax_a.plot([min_val, max_val], [min_val, max_val], linestyle="--", label="Línea ideal")
    ax_a.set_xlabel(f"Valor real ({TARGET_COL})")
    ax_a.set_ylabel(f"Predicción ({TARGET_COL})")
    ax_a.set_title("Real vs Predicho – Árbol de Decisión")
    ax_a.legend()
    imagenes["real_vs_pred"] = fig_to_base64(fig_a)
    plt.close(fig_a)

    # --- Gráfico B: Estructura del árbol (primeros niveles) ---
    fig_b, ax_b = plt.subplots(figsize=(18, 8))
    plot_tree(
        tree_reg,
        feature_names=feature_cols,
        filled=True,
        max_depth=3,
        rounded=True,
        ax=ax_b,
    )
    ax_b.set_title("Estructura del Árbol de Decisión (primeros niveles)")
    imagenes["arbol"] = fig_to_base64(fig_b)
    plt.close(fig_b)

    # --- Gráfico C: Importancia de variables ---
    importances = tree_reg.feature_importances_
    imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values(
        "importance", ascending=False
    )

    fig_c, ax_c = plt.subplots(figsize=(8, 5))
    ax_c.bar(imp_df["feature"], imp_df["importance"])
    ax_c.set_xticklabels(imp_df["feature"], rotation=90)
    ax_c.set_title("Importancia de variables – Árbol de Decisión")
    fig_c.tight_layout()
    imagenes["importancias"] = fig_to_base64(fig_c)
    plt.close(fig_c)

    metricas = {
        "mse": round(mse_tree, 3),
        "mae": round(mae_tree, 3),
        "r2": round(r2_tree, 4),
        "profundidad": tree_reg.get_depth(),
        "n_nodos": tree_reg.tree_.node_count,
    }

    return imagenes, metricas, None