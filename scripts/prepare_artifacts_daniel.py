"""
Genera los artefactos auxiliares del modelo de rezago por área (Daniel).

Reproduce el split train/test del notebook, evalúa los 5 modelos Keras y los
5 pipelines HGB sobre el test, y guarda en models/daniel/ los archivos que el
dashboard solo tiene que leer:

  - metricas_test.json          métricas recalculadas (sanity vs metadata)
  - curvas_roc.json             fpr/tpr/thresholds + AUC por área y modelo
  - curvas_pr.json              precision/recall/thresholds + AP por área y modelo
  - matrices_confusion.json     matriz a umbral 0.5 por área y modelo
  - importancia_hgb.csv         feature importances del HGB benchmark por área
  - comparacion_modelos.csv     tabla resumen Keras vs HGB por área

Se ejecuta una sola vez. El dashboard nunca recalcula nada de esto.
"""

import json
import os
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.2
N_PUNTOS_CURVA = 200
N_PERM_REPEATS = 5
N_PERM_SAMPLE = 5000

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "DatosSaber11_Bolivar_limpio_todas_columnas.csv"
MODELS_DIR = ROOT / "models" / "daniel"
META_PATH = MODELS_DIR / "metadata_rezago_areas.json"


def cargar_metadata():
    with open(META_PATH, encoding="utf-8") as f:
        return json.load(f)


def construir_df_modelo(df, predictores, areas):
    cols_numericas = [c for c in ["edad", "periodo"] if c in predictores]
    cols_categoricas = [c for c in predictores if c not in cols_numericas]

    df_modelo = df[predictores + [a["puntaje_origen"] for a in areas]].copy()
    for c in cols_numericas:
        df_modelo[c] = pd.to_numeric(df_modelo[c], errors="coerce")
    for c in cols_categoricas:
        df_modelo[c] = df_modelo[c].astype(str)
    return df_modelo


def calcular_targets_rezago(df_modelo, areas, idx_train):
    targets = {}
    for area in areas:
        col_punt = area["puntaje_origen"]
        p25 = df_modelo.iloc[idx_train][col_punt].quantile(0.25)
        target = (df_modelo[col_punt] < p25).astype(int)
        targets[area["area"]] = {
            "y": target.values,
            "p25_train": float(p25),
            "p25_metadata": area["percentil_25_train"],
        }
    return targets


def submuestrear_curva(arr_x, arr_y, n=N_PUNTOS_CURVA):
    if len(arr_x) <= n:
        return arr_x.tolist(), arr_y.tolist()
    idx = np.linspace(0, len(arr_x) - 1, n).astype(int)
    return arr_x[idx].tolist(), arr_y[idx].tolist()


def evaluar_area(area, modelo_keras, pipeline_hgb, X_test_raw, X_test_trans, y_test):
    proba_keras = modelo_keras.predict(X_test_trans, verbose=0).ravel()
    proba_hgb = pipeline_hgb.predict_proba(X_test_raw)[:, 1]

    pred_keras = (proba_keras >= 0.5).astype(int)
    pred_hgb = (proba_hgb >= 0.5).astype(int)

    metricas = {}
    curvas_roc = {}
    curvas_pr = {}
    matrices = {}

    for nombre, proba, pred in [
        ("keras", proba_keras, pred_keras),
        ("hgb", proba_hgb, pred_hgb),
    ]:
        metricas[nombre] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "average_precision": float(average_precision_score(y_test, proba)),
        }

        fpr, tpr, _ = roc_curve(y_test, proba)
        fpr_s, tpr_s = submuestrear_curva(fpr, tpr)
        curvas_roc[nombre] = {
            "fpr": fpr_s, "tpr": tpr_s,
            "auc": metricas[nombre]["roc_auc"],
        }

        prec, rec, _ = precision_recall_curve(y_test, proba)
        rec_s, prec_s = submuestrear_curva(rec, prec)
        curvas_pr[nombre] = {
            "recall": rec_s, "precision": prec_s,
            "ap": metricas[nombre]["average_precision"],
        }

        cm = confusion_matrix(y_test, pred)
        matrices[nombre] = {
            "matriz": cm.tolist(),
            "threshold": 0.5,
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        }

    return metricas, curvas_roc, curvas_pr, matrices


def calcular_importancia_permutation(pipeline_hgb, X_sample, y_sample, area_nombre, predictores):
    resultado = permutation_importance(
        pipeline_hgb, X_sample, y_sample,
        n_repeats=N_PERM_REPEATS, random_state=SEED, scoring="roc_auc",
        n_jobs=-1,
    )
    df_imp = pd.DataFrame({
        "feature": predictores,
        "importancia_media": resultado.importances_mean,
        "importancia_std": resultado.importances_std,
        "area": area_nombre,
    }).sort_values("importancia_media", ascending=False).reset_index(drop=True)
    return df_imp


def main():
    print("Leyendo metadata y CSV...")
    meta = cargar_metadata()
    df = pd.read_csv(DATA_PATH)
    print(f"  CSV: {df.shape[0]:,} filas × {df.shape[1]} columnas")

    predictores = meta["predictores"]
    areas = meta["areas_modeladas"]

    df_modelo = construir_df_modelo(df, predictores, areas)

    print("Reproduciendo split train/test...")
    idx_train, idx_test = train_test_split(
        np.arange(len(df_modelo)),
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=True,
    )
    print(f"  Train: {len(idx_train):,}   Test: {len(idx_test):,}")
    if len(idx_train) != meta["particion"]["n_train"]:
        print(f"  AVISO: n_train reproducido={len(idx_train)} vs metadata={meta['particion']['n_train']}")

    targets = calcular_targets_rezago(df_modelo, areas, idx_train)

    print("Cargando preprocessor y transformando X_test...")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor_rezago_areas.joblib")
    X_test_raw = df_modelo.iloc[idx_test][predictores].copy()
    X_test_trans = preprocessor.transform(X_test_raw)
    print(f"  X_test transformado: {X_test_trans.shape}")

    metricas_globales = {}
    curvas_roc_globales = {}
    curvas_pr_globales = {}
    matrices_globales = {}
    importancias = []
    comparacion = []

    rng_sample = np.random.default_rng(SEED)
    idx_sample = rng_sample.choice(len(idx_test), size=min(N_PERM_SAMPLE, len(idx_test)), replace=False)
    X_sample = X_test_raw.iloc[idx_sample]

    for area in areas:
        nombre = area["area"]
        print(f"\nÁrea: {nombre}")

        modelo_keras = tf.keras.models.load_model(
            MODELS_DIR / area["archivo_modelo_keras"], compile=False
        )
        pipeline_hgb = joblib.load(MODELS_DIR / area["archivo_modelo_benchmark_ganador"])

        y_test = targets[nombre]["y"][idx_test]
        y_sample = y_test[idx_sample]

        metricas, curvas_roc, curvas_pr, matrices = evaluar_area(
            area, modelo_keras, pipeline_hgb, X_test_raw, X_test_trans, y_test
        )

        metricas_globales[nombre] = metricas
        curvas_roc_globales[nombre] = curvas_roc
        curvas_pr_globales[nombre] = curvas_pr
        matrices_globales[nombre] = matrices

        keras_auc_actual = metricas["keras"]["roc_auc"]
        keras_auc_meta = area["metricas_keras"]["roc_auc"]
        print(f"  Keras AUC reproducido={keras_auc_actual:.4f}  metadata={keras_auc_meta:.4f}  Δ={abs(keras_auc_actual-keras_auc_meta):.4f}")

        hgb_auc_actual = metricas["hgb"]["roc_auc"]
        hgb_auc_meta = area["metricas_benchmark_ganador"]["roc_auc"]
        print(f"  HGB   AUC reproducido={hgb_auc_actual:.4f}  metadata={hgb_auc_meta:.4f}  Δ={abs(hgb_auc_actual-hgb_auc_meta):.4f}")

        print(f"  Calculando permutation importance (HGB, n={len(idx_sample)}, repeats={N_PERM_REPEATS})...")
        df_imp = calcular_importancia_permutation(
            pipeline_hgb, X_sample, y_sample, nombre, predictores
        )
        importancias.append(df_imp)
        top3 = df_imp.head(3)["feature"].tolist()
        print(f"  Top-3 importancia: {top3}")

        for modelo_n in ["keras", "hgb"]:
            comparacion.append({
                "area": nombre,
                "modelo": modelo_n.upper(),
                **{k: round(v, 4) for k, v in metricas[modelo_n].items()},
            })

    print("\nGuardando artefactos...")
    with open(MODELS_DIR / "metricas_test.json", "w", encoding="utf-8") as f:
        json.dump(metricas_globales, f, indent=2, ensure_ascii=False)
    with open(MODELS_DIR / "curvas_roc.json", "w", encoding="utf-8") as f:
        json.dump(curvas_roc_globales, f, ensure_ascii=False)
    with open(MODELS_DIR / "curvas_pr.json", "w", encoding="utf-8") as f:
        json.dump(curvas_pr_globales, f, ensure_ascii=False)
    with open(MODELS_DIR / "matrices_confusion.json", "w", encoding="utf-8") as f:
        json.dump(matrices_globales, f, indent=2, ensure_ascii=False)

    if importancias:
        pd.concat(importancias, ignore_index=True).to_csv(
            MODELS_DIR / "importancia_permutation.csv", index=False, encoding="utf-8"
        )
    pd.DataFrame(comparacion).to_csv(
        MODELS_DIR / "comparacion_modelos.csv", index=False, encoding="utf-8"
    )

    print(f"\nListo. Artefactos en {MODELS_DIR}")
    for archivo in [
        "metricas_test.json", "curvas_roc.json", "curvas_pr.json",
        "matrices_confusion.json", "importancia_permutation.csv", "comparacion_modelos.csv",
    ]:
        path = MODELS_DIR / archivo
        if path.exists():
            print(f"  {archivo:30s}  {path.stat().st_size/1024:>8.1f} KB")


if __name__ == "__main__":
    main()
