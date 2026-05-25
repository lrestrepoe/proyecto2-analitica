"""
Genera los artefactos del modelo de rezago por área (Daniel).

Flujo:
  1. Reproduce el split train/test con el preprocessor ya guardado.
  2. Para cada área entrena 5 configuraciones de red neuronal.
  3. Selecciona la mejor configuración por AUC (F1 desempate).
  4. Guarda la mejor como keras_rezago_<area>.keras (sobrescribe la anterior).
  5. Recalcula y guarda curvas, matrices, métricas e importancia.
  6. (Opcional) registra cada corrida en MLflow si está instalado.

Variables de control (entorno):
  EPOCHS_MAX (default 15)
  PATIENCE   (default 4)
  N_PERM_SAMPLE (default 5000)
  N_PERM_REPEATS (default 5)
"""

import json
import os
import time
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
from tensorflow.keras import layers, optimizers, regularizers

SEED = 42
TEST_SIZE = 0.2
N_PUNTOS_CURVA = 200
EPOCHS_MAX = int(os.environ.get("EPOCHS_MAX", 15))
PATIENCE = int(os.environ.get("PATIENCE", 4))
N_PERM_REPEATS = int(os.environ.get("N_PERM_REPEATS", 5))
N_PERM_SAMPLE = int(os.environ.get("N_PERM_SAMPLE", 5000))
VAL_FRAC = 0.15

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "DatosSaber11_Bolivar_limpio_todas_columnas.csv"
MODELS_DIR = ROOT / "models" / "daniel"
META_PATH = MODELS_DIR / "metadata_rezago_areas.json"

tf.random.set_seed(SEED)
np.random.seed(SEED)


CONFIGURACIONES = [
    {"nombre": "Keras_32x16",           "capas": [32, 16],   "dropout": 0.0, "lr": 0.001,  "batch_size": 32},
    {"nombre": "Keras_64x32",           "capas": [64, 32],   "dropout": 0.0, "lr": 0.001,  "batch_size": 32},
    {"nombre": "Keras_128x64_dropout",  "capas": [128, 64],  "dropout": 0.3, "lr": 0.001,  "batch_size": 32},
    {"nombre": "Keras_64x32_lr0005",    "capas": [64, 32],   "dropout": 0.0, "lr": 0.0005, "batch_size": 32},
    {"nombre": "Keras_64x32_batch64",   "capas": [64, 32],   "dropout": 0.0, "lr": 0.001,  "batch_size": 64},
]


def construir_modelo(input_dim, capas, dropout, lr):
    tf.keras.utils.set_random_seed(SEED)
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    for unidades in capas:
        x = layers.Dense(unidades, activation="relu",
                          kernel_regularizer=regularizers.l2(1e-5))(x)
        if dropout > 0:
            x = layers.Dropout(dropout)(x)
    output = layers.Dense(1, activation="sigmoid")(x)
    modelo = tf.keras.Model(inputs, output)
    modelo.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


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
        targets[area["area"]] = {"y": target.values, "p25_train": float(p25)}
    return targets


def submuestrear_curva(arr_x, arr_y, n=N_PUNTOS_CURVA):
    if len(arr_x) <= n:
        return arr_x.tolist(), arr_y.tolist()
    idx = np.linspace(0, len(arr_x) - 1, n).astype(int)
    return arr_x[idx].tolist(), arr_y[idx].tolist()


def evaluar_proba(y_true, proba):
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "average_precision": float(average_precision_score(y_true, proba)),
    }


def entrenar_configuracion(config, X_train, y_train, X_val, y_val,
                            X_test, y_test, mlflow_mod=None, area_nombre=""):
    modelo = construir_modelo(X_train.shape[1], config["capas"],
                                config["dropout"], config["lr"])
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=0,
    )
    t0 = time.time()
    historial = modelo.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS_MAX,
        batch_size=config["batch_size"],
        callbacks=[early],
        verbose=0,
    )
    duracion = time.time() - t0
    epochs_entrenadas = len(historial.history["loss"])
    proba_test = modelo.predict(X_test, verbose=0).ravel()
    metricas = evaluar_proba(y_test, proba_test)

    fila = {
        "area": area_nombre,
        "configuracion": config["nombre"],
        "arquitectura": "+".join(map(str, config["capas"])),
        "dropout": config["dropout"],
        "learning_rate": config["lr"],
        "batch_size": config["batch_size"],
        "epochs_entrenadas": epochs_entrenadas,
        "duracion_seg": round(duracion, 1),
        **{k: round(v, 4) for k, v in metricas.items()},
    }

    if mlflow_mod is not None:
        try:
            with mlflow_mod.start_run(run_name=f"{area_nombre}_{config['nombre']}"):
                mlflow_mod.log_params({
                    "area": area_nombre, "configuracion": config["nombre"],
                    "arquitectura": fila["arquitectura"],
                    "dropout": config["dropout"],
                    "learning_rate": config["lr"],
                    "batch_size": config["batch_size"],
                    "epochs_max": EPOCHS_MAX, "patience": PATIENCE,
                })
                mlflow_mod.log_metrics({
                    "epochs_entrenadas": epochs_entrenadas,
                    "duracion_seg": duracion,
                    **metricas,
                })
        except Exception as e:
            print(f"  [MLflow warning] {e}")

    return modelo, proba_test, fila


def seleccionar_mejor(filas_area):
    df = pd.DataFrame(filas_area)
    df_sorted = df.sort_values(
        ["roc_auc", "f1", "recall"], ascending=[False, False, False]
    )
    return df_sorted.iloc[0].to_dict()


def calcular_importancia_permutation(pipeline_hgb, X_sample, y_sample, area_nombre, predictores):
    resultado = permutation_importance(
        pipeline_hgb, X_sample, y_sample,
        n_repeats=N_PERM_REPEATS, random_state=SEED, scoring="roc_auc",
        n_jobs=-1,
    )
    return pd.DataFrame({
        "feature": predictores,
        "importancia_media": resultado.importances_mean,
        "importancia_std": resultado.importances_std,
        "area": area_nombre,
    }).sort_values("importancia_media", ascending=False).reset_index(drop=True)


def main():
    print("=" * 70)
    print("Generación de artefactos de Daniel - Comparación de configuraciones Keras")
    print("=" * 70)
    print(f"  EPOCHS_MAX={EPOCHS_MAX}  PATIENCE={PATIENCE}  SEED={SEED}")

    mlflow_mod = None
    try:
        import mlflow as _mlflow
        _mlflow.set_experiment("rezago_areas_daniel")
        mlflow_mod = _mlflow
        print("  MLflow activado")
    except Exception as e:
        print(f"  MLflow desactivado ({type(e).__name__}). Continuando sin tracking.")

    print("\nLeyendo metadata y CSV...")
    meta = cargar_metadata()
    df = pd.read_csv(DATA_PATH)
    print(f"  CSV: {df.shape[0]:,} filas × {df.shape[1]} columnas")

    predictores = meta["predictores"]
    areas = meta["areas_modeladas"]
    df_modelo = construir_df_modelo(df, predictores, areas)

    idx_train, idx_test = train_test_split(
        np.arange(len(df_modelo)), test_size=TEST_SIZE,
        random_state=SEED, shuffle=True,
    )
    print(f"  Train+Val: {len(idx_train):,}   Test: {len(idx_test):,}")

    targets = calcular_targets_rezago(df_modelo, areas, idx_train)

    print("\nCargando preprocessor y transformando X...")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor_rezago_areas.joblib")
    X_train_raw = df_modelo.iloc[idx_train][predictores]
    X_test_raw = df_modelo.iloc[idx_test][predictores]
    X_train_full = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    if hasattr(X_train_full, "toarray"):
        X_train_full = X_train_full.toarray()
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()
    X_train_full = X_train_full.astype(np.float32)
    X_test = X_test.astype(np.float32)
    print(f"  X_train_full: {X_train_full.shape}   X_test: {X_test.shape}")

    rng = np.random.default_rng(SEED)
    idx_sample = rng.choice(len(idx_test), size=min(N_PERM_SAMPLE, len(idx_test)), replace=False)
    X_perm_sample = X_test_raw.iloc[idx_sample]

    todas_configuraciones = []
    mejores_por_area = []
    metricas_globales = {}
    curvas_roc_globales = {}
    curvas_pr_globales = {}
    matrices_globales = {}
    importancias = []
    comparacion_final = []
    metadata_areas_actualizado = []

    t_inicio = time.time()

    for i, area in enumerate(areas, 1):
        nombre = area["area"]
        archivo_keras = area["archivo_modelo_keras"]
        print(f"\n[{i}/{len(areas)}] Área: {nombre}")

        y_full = targets[nombre]["y"][idx_train]
        y_test = targets[nombre]["y"][idx_test]

        n_val = int(len(idx_train) * VAL_FRAC)
        idx_perm = rng.permutation(len(idx_train))
        idx_tr, idx_val = idx_perm[n_val:], idx_perm[:n_val]
        X_tr, X_val = X_train_full[idx_tr], X_train_full[idx_val]
        y_tr, y_val = y_full[idx_tr], y_full[idx_val]

        modelos_area = {}
        filas_area = []
        for config in CONFIGURACIONES:
            print(f"    Entrenando {config['nombre']:25s} ...", end="", flush=True)
            t0 = time.time()
            modelo, proba_test, fila = entrenar_configuracion(
                config, X_tr, y_tr, X_val, y_val, X_test, y_test,
                mlflow_mod=mlflow_mod, area_nombre=nombre,
            )
            modelos_area[config["nombre"]] = (modelo, proba_test)
            filas_area.append(fila)
            print(f" AUC={fila['roc_auc']:.4f}  F1={fila['f1']:.4f}  "
                   f"epochs={fila['epochs_entrenadas']:2d}  ({time.time()-t0:.1f}s)")
            todas_configuraciones.append(fila)

        mejor = seleccionar_mejor(filas_area)
        mejor_nombre = mejor["configuracion"]
        print(f"  → Mejor configuración: {mejor_nombre}  (AUC={mejor['roc_auc']:.4f})")
        mejores_por_area.append({
            "area": nombre,
            "mejor_configuracion": mejor_nombre,
            "roc_auc": mejor["roc_auc"], "f1": mejor["f1"],
            "recall": mejor["recall"], "precision": mejor["precision"],
            "accuracy": mejor["accuracy"],
            "average_precision": mejor["average_precision"],
        })

        modelo_mejor, proba_keras = modelos_area[mejor_nombre]
        modelo_mejor.save(MODELS_DIR / archivo_keras)
        print(f"  ✓ Guardado {archivo_keras}")

        pipeline_hgb = joblib.load(MODELS_DIR / area["archivo_modelo_benchmark_ganador"])
        proba_hgb = pipeline_hgb.predict_proba(X_test_raw)[:, 1]
        metricas_hgb = evaluar_proba(y_test, proba_hgb)
        metricas_keras = {k: mejor[k] for k in
                           ["accuracy", "precision", "recall", "f1",
                            "roc_auc", "average_precision"]}

        metricas_globales[nombre] = {"keras": metricas_keras, "hgb": metricas_hgb}

        fpr_k, tpr_k, _ = roc_curve(y_test, proba_keras)
        fpr_h, tpr_h, _ = roc_curve(y_test, proba_hgb)
        fpr_ks, tpr_ks = submuestrear_curva(fpr_k, tpr_k)
        fpr_hs, tpr_hs = submuestrear_curva(fpr_h, tpr_h)
        curvas_roc_globales[nombre] = {
            "keras": {"fpr": fpr_ks, "tpr": tpr_ks, "auc": metricas_keras["roc_auc"]},
            "hgb":   {"fpr": fpr_hs, "tpr": tpr_hs, "auc": metricas_hgb["roc_auc"]},
        }

        prec_k, rec_k, _ = precision_recall_curve(y_test, proba_keras)
        prec_h, rec_h, _ = precision_recall_curve(y_test, proba_hgb)
        rec_ks, prec_ks = submuestrear_curva(rec_k, prec_k)
        rec_hs, prec_hs = submuestrear_curva(rec_h, prec_h)
        curvas_pr_globales[nombre] = {
            "keras": {"recall": rec_ks, "precision": prec_ks, "ap": metricas_keras["average_precision"]},
            "hgb":   {"recall": rec_hs, "precision": prec_hs, "ap": metricas_hgb["average_precision"]},
        }

        pred_k = (proba_keras >= 0.5).astype(int)
        pred_h = (proba_hgb >= 0.5).astype(int)
        cm_k = confusion_matrix(y_test, pred_k)
        cm_h = confusion_matrix(y_test, pred_h)
        matrices_globales[nombre] = {
            "keras": {"matriz": cm_k.tolist(), "threshold": 0.5,
                       "tn": int(cm_k[0,0]), "fp": int(cm_k[0,1]),
                       "fn": int(cm_k[1,0]), "tp": int(cm_k[1,1])},
            "hgb":   {"matriz": cm_h.tolist(), "threshold": 0.5,
                       "tn": int(cm_h[0,0]), "fp": int(cm_h[0,1]),
                       "fn": int(cm_h[1,0]), "tp": int(cm_h[1,1])},
        }

        y_sample = y_test[idx_sample]
        df_imp = calcular_importancia_permutation(
            pipeline_hgb, X_perm_sample, y_sample, nombre, predictores,
        )
        importancias.append(df_imp)

        for modelo_n, m in [("KERAS", metricas_keras), ("HGB", metricas_hgb)]:
            comparacion_final.append({
                "area": nombre, "modelo": modelo_n,
                **{k: round(v, 4) for k, v in m.items()},
            })

        area_actualizada = dict(area)
        area_actualizada["metricas_keras"] = metricas_keras
        area_actualizada["metricas_benchmark_ganador"] = metricas_hgb
        area_actualizada["mejor_configuracion_keras"] = mejor_nombre
        metadata_areas_actualizado.append(area_actualizada)

    t_total = time.time() - t_inicio
    print(f"\nTiempo total de entrenamiento: {t_total/60:.1f} min")

    print("\nGuardando artefactos...")

    df_configs = pd.DataFrame(todas_configuraciones)
    df_configs = df_configs[[
        "area", "configuracion", "accuracy", "precision", "recall", "f1",
        "roc_auc", "average_precision", "epochs_entrenadas", "batch_size",
        "learning_rate", "arquitectura", "dropout", "duracion_seg",
    ]]
    df_configs.to_csv(MODELS_DIR / "comparacion_configuraciones_keras.csv",
                       index=False, encoding="utf-8")

    pd.DataFrame(mejores_por_area).to_csv(
        MODELS_DIR / "mejores_configuraciones_keras.csv",
        index=False, encoding="utf-8",
    )

    pd.DataFrame(comparacion_final).to_csv(
        MODELS_DIR / "comparacion_modelos.csv",
        index=False, encoding="utf-8",
    )

    with open(MODELS_DIR / "metricas_test.json", "w", encoding="utf-8") as f:
        json.dump(metricas_globales, f, indent=2, ensure_ascii=False)
    with open(MODELS_DIR / "curvas_roc.json", "w", encoding="utf-8") as f:
        json.dump(curvas_roc_globales, f, ensure_ascii=False)
    with open(MODELS_DIR / "curvas_pr.json", "w", encoding="utf-8") as f:
        json.dump(curvas_pr_globales, f, ensure_ascii=False)
    with open(MODELS_DIR / "matrices_confusion.json", "w", encoding="utf-8") as f:
        json.dump(matrices_globales, f, indent=2, ensure_ascii=False)

    pd.concat(importancias, ignore_index=True).to_csv(
        MODELS_DIR / "importancia_permutation.csv",
        index=False, encoding="utf-8",
    )

    meta["areas_modeladas"] = metadata_areas_actualizado
    meta["mejores_configuraciones"] = {
        m["area"]: m["mejor_configuracion"] for m in mejores_por_area
    }
    meta["configuraciones_evaluadas"] = CONFIGURACIONES
    meta["entrenamiento"] = {
        "epochs_max": EPOCHS_MAX, "patience": PATIENCE,
        "seed": SEED, "val_frac": VAL_FRAC,
        "tiempo_total_seg": round(t_total, 1),
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n  Artefactos en {MODELS_DIR}:")
    for archivo in [
        "comparacion_configuraciones_keras.csv",
        "mejores_configuraciones_keras.csv",
        "comparacion_modelos.csv",
        "metricas_test.json", "curvas_roc.json", "curvas_pr.json",
        "matrices_confusion.json", "importancia_permutation.csv",
        "metadata_rezago_areas.json",
    ]:
        path = MODELS_DIR / archivo
        if path.exists():
            print(f"    {archivo:42s}  {path.stat().st_size/1024:>8.1f} KB")

    print("\n=== AUC máximo por área ===")
    print(df_configs.groupby("area")["roc_auc"].max().round(4).to_string())


if __name__ == "__main__":
    main()
