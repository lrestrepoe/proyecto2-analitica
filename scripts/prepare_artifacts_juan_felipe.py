"""
Genera los artefactos auxiliares del modelo de regresión de puntaje global (Juan Felipe).

Reproduce el preprocesamiento y el split del notebook, evalúa el modelo Keras
sobre el test, y guarda en models/juan_felipe/ los archivos que el dashboard
solo tiene que leer:

  - metricas_test.json              MAE, RMSE, R², MAPE recalculados
  - predicciones_vs_reales.csv      sample de 5k filas (real, predicho, residuo)
  - distribucion_residuos.json      histograma precomputado de residuos
  - importancia_permutation.csv     permutation importance sobre el test
  - comparacion_observaciones.csv   estadísticos por subgrupo (zona, estrato, etc.)

Se ejecuta una sola vez. El dashboard nunca recalcula nada de esto.
"""

import json
import os
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
TEST_FRAC = 0.8
N_PRED_SAMPLE = 5000
N_PERM_REPEATS = 5
N_PERM_SAMPLE = 5000
N_BINS_HIST = 50

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "DatosSaber11_Bolivar_limpio_todas_columnas.csv"
MODELS_DIR = ROOT / "models" / "juan_felipe"
ARTEFACTOS_PATH = MODELS_DIR / "artefactos.json"
MODELO_PATH = MODELS_DIR / "modelo_saber11.keras"

FEATURES_BINARIAS = [
    "cole_bilingue", "cole_area_ubicacion", "cole_naturaleza",
    "fami_tieneautomovil", "fami_tienecomputador", "fami_tieneinternet",
    "fami_tienelavadora", "estu_genero",
]
FEATURES_ORDINALES = [
    "fami_estratovivienda", "fami_educacionmadre", "fami_educacionpadre",
    "fami_cuartoshogar", "fami_personashogar",
]
FEATURES_NOMINALES = ["cole_calendario", "cole_caracter", "cole_jornada", "cole_genero"]
FEATURES_NUMERICAS = ["edad"]
TARGET = "punt_global"

TODAS_FEATURES = FEATURES_BINARIAS + FEATURES_ORDINALES + FEATURES_NOMINALES + FEATURES_NUMERICAS


def cargar_artefactos():
    with open(ARTEFACTOS_PATH, encoding="utf-8") as f:
        return json.load(f)


def aplicar_mapeos_binarios(df, artefactos):
    df = df.copy()
    df["cole_bilingue"] = df["cole_bilingue"].map(artefactos["mapeo_bilingue"])
    df["cole_area_ubicacion"] = df["cole_area_ubicacion"].map(artefactos["mapeo_area"])
    df["cole_naturaleza"] = df["cole_naturaleza"].map(artefactos["mapeo_naturaleza"])
    df["fami_tieneautomovil"] = df["fami_tieneautomovil"].map(artefactos["mapeo_si_no"])
    df["fami_tienecomputador"] = df["fami_tienecomputador"].map(artefactos["mapeo_si_no"])
    df["fami_tieneinternet"] = df["fami_tieneinternet"].map(artefactos["mapeo_si_no"])
    df["fami_tienelavadora"] = df["fami_tienelavadora"].map(artefactos["mapeo_si_no"])
    df["estu_genero"] = df["estu_genero"].map(artefactos["mapeo_genero_estu"])
    return df


def aplicar_mapeos_ordinales(df, artefactos):
    df = df.copy()
    df["fami_estratovivienda"] = df["fami_estratovivienda"].map(artefactos["mapeo_estrato"])
    df["fami_educacionmadre"] = df["fami_educacionmadre"].map(artefactos["mapeo_educacion"])
    df["fami_educacionpadre"] = df["fami_educacionpadre"].map(artefactos["mapeo_educacion"])
    df["fami_cuartoshogar"] = df["fami_cuartoshogar"].map(artefactos["mapeo_cuartos"])
    df["fami_personashogar"] = df["fami_personashogar"].map(artefactos["mapeo_personas"])
    return df


def construir_dummies(df, artefactos):
    df = pd.get_dummies(df, columns=FEATURES_NOMINALES, prefix=FEATURES_NOMINALES, dtype=int)
    columnas_modelo = artefactos["columnas_modelo"]
    for col in columnas_modelo:
        if col not in df.columns:
            df[col] = 0
    return df[columnas_modelo + [TARGET]]


def preprocesar_df(df, artefactos):
    df_m = df[TODAS_FEATURES + [TARGET]].copy()
    df_m = aplicar_mapeos_binarios(df_m, artefactos)
    df_m = aplicar_mapeos_ordinales(df_m, artefactos)
    df_m = construir_dummies(df_m, artefactos)
    return df_m


def reproducir_split(df_modelo):
    np.random.seed(SEED)
    train_df = df_modelo.sample(frac=TEST_FRAC, random_state=SEED)
    test_df = df_modelo.drop(train_df.index)
    return train_df, test_df


def predict_wrapper(modelo, X):
    return modelo.predict(X.astype(np.float32), verbose=0).ravel()


def main():
    print("Leyendo artefactos y CSV...")
    artefactos = cargar_artefactos()
    df = pd.read_csv(DATA_PATH)
    print(f"  CSV: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    print(f"  Columnas modelo esperadas: {len(artefactos['columnas_modelo'])}")

    print("\nAplicando preprocesamiento del notebook...")
    df_modelo = preprocesar_df(df, artefactos)
    nulos = df_modelo.isna().sum().sum()
    print(f"  Shape final: {df_modelo.shape}   Nulos: {nulos}")
    if nulos > 0:
        print("  AVISO: hay nulos después del mapeo. Revisar categorías no contempladas.")

    print("\nReproduciendo split (sample frac=0.8, random_state=42)...")
    train_df, test_df = reproducir_split(df_modelo)
    print(f"  Train: {train_df.shape[0]:,}   Test: {test_df.shape[0]:,}")

    test_X = test_df.drop(columns=[TARGET]).copy()
    test_y = test_df[TARGET].values

    print("\nCargando modelo Keras...")
    modelo = tf.keras.models.load_model(MODELO_PATH, compile=False)
    print(f"  Input shape esperado: {modelo.input_shape}")
    print(f"  Columnas reales en test_X: {test_X.shape[1]}")

    print("\nPrediciendo sobre test...")
    pred = predict_wrapper(modelo, test_X.values)
    residuos = test_y - pred

    mae = float(mean_absolute_error(test_y, pred))
    rmse = float(np.sqrt(mean_squared_error(test_y, pred)))
    r2 = float(r2_score(test_y, pred))
    mask_nz = test_y != 0
    mape = float(np.mean(np.abs(residuos[mask_nz] / test_y[mask_nz])) * 100)

    metricas = {
        "n_train": int(train_df.shape[0]),
        "n_test": int(test_df.shape[0]),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "mape": round(mape, 4),
        "residuo_media": round(float(residuos.mean()), 4),
        "residuo_std": round(float(residuos.std()), 4),
        "real_min": float(test_y.min()),
        "real_max": float(test_y.max()),
        "pred_min": round(float(pred.min()), 2),
        "pred_max": round(float(pred.max()), 2),
        "mae_metadata": artefactos["metricas"]["MAE"],
        "r2_metadata": artefactos["metricas"]["R2"],
    }
    print(f"\n  MAE reproducido: {mae:.4f}  metadata: {artefactos['metricas']['MAE']}")
    print(f"  R²  reproducido: {r2:.4f}  metadata: {artefactos['metricas']['R2']}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAPE: {mape:.2f}%")

    print(f"\nGuardando sample de predicciones (n={N_PRED_SAMPLE})...")
    rng = np.random.default_rng(SEED)
    idx_sample = rng.choice(len(test_y), size=min(N_PRED_SAMPLE, len(test_y)), replace=False)

    df_preds = pd.DataFrame({
        "puntaje_real": test_y[idx_sample].astype(float),
        "puntaje_predicho": pred[idx_sample].astype(float).round(2),
        "residuo": residuos[idx_sample].astype(float).round(2),
    })
    df_preds.to_csv(MODELS_DIR / "predicciones_vs_reales.csv", index=False, encoding="utf-8")

    print("Guardando histograma de residuos precomputado...")
    counts, bin_edges = np.histogram(residuos, bins=N_BINS_HIST)
    hist = {
        "bin_edges": bin_edges.tolist(),
        "counts": counts.tolist(),
        "n_total": int(len(residuos)),
        "media": round(float(residuos.mean()), 4),
        "std": round(float(residuos.std()), 4),
        "p05": round(float(np.quantile(residuos, 0.05)), 4),
        "p95": round(float(np.quantile(residuos, 0.95)), 4),
    }
    with open(MODELS_DIR / "distribucion_residuos.json", "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2, ensure_ascii=False)

    print(f"\nCalculando permutation importance (n={N_PERM_SAMPLE}, repeats={N_PERM_REPEATS})...")
    idx_perm = rng.choice(len(test_y), size=min(N_PERM_SAMPLE, len(test_y)), replace=False)
    X_perm = test_X.iloc[idx_perm].values.astype(np.float32)
    y_perm = test_y[idx_perm]

    class KerasRegresorWrapper(RegressorMixin, BaseEstimator):
        def __init__(self, modelo=None):
            self.modelo = modelo
        def predict(self, X):
            return predict_wrapper(self.modelo, np.asarray(X))
        def fit(self, X, y):
            self.is_fitted_ = True
            return self
        def __sklearn_is_fitted__(self):
            return True

    wrapper = KerasRegresorWrapper(modelo=modelo)
    resultado = permutation_importance(
        wrapper, X_perm, y_perm,
        n_repeats=N_PERM_REPEATS, random_state=SEED,
        scoring="neg_mean_absolute_error",
    )
    df_imp = pd.DataFrame({
        "feature": test_X.columns.tolist(),
        "importancia_media": resultado.importances_mean,
        "importancia_std": resultado.importances_std,
    }).sort_values("importancia_media", ascending=False).reset_index(drop=True)
    df_imp.to_csv(MODELS_DIR / "importancia_permutation.csv", index=False, encoding="utf-8")
    print(f"  Top-5: {df_imp.head(5)['feature'].tolist()}")

    print("\nGuardando métricas...")
    with open(MODELS_DIR / "metricas_test.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)

    print(f"\nListo. Artefactos en {MODELS_DIR}")
    for archivo in [
        "metricas_test.json", "predicciones_vs_reales.csv",
        "distribucion_residuos.json", "importancia_permutation.csv",
    ]:
        path = MODELS_DIR / archivo
        if path.exists():
            print(f"  {archivo:32s}  {path.stat().st_size/1024:>8.1f} KB")


if __name__ == "__main__":
    main()
