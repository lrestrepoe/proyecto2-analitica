"""
Genera los artefactos auxiliares del modelo de prediccion de zona urbana/rural (Luis).

Reproduce el split train/test del notebook, evalua el modelo Keras sobre el test,
y guarda en models/luis/ los archivos que el dashboard solo tiene que leer:

  - metricas_test.json          accuracy, auc, recall, precision, f1
  - matriz_confusion.json       matriz de confusion a umbral 0.5
  - curvas_roc.json             fpr/tpr + AUC
  - comparacion_modelos.csv     tabla resumen E1, E2, E3
  - metadata_luis.json          descripcion del modelo y variables

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
import keras
from sklearn.metrics import (
    accuracy_score, roc_auc_score, recall_score,
    precision_score, f1_score, confusion_matrix,
    roc_curve, auc
)

SEED = 42
ROOT      = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "DatosSaber11_Bolivar_limpio_todas_columnas.csv"
MODEL_DIR = ROOT / "models" / "luis"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "cole_area_ubicacion"

cat_str_feats = ["estu_genero", "fami_estratovivienda", "fami_educacionpadre",
                 "fami_educacionmadre", "fami_tieneinternet", "fami_tienecomputador",
                 "fami_tieneautomovil", "fami_tienelavadora", "cole_naturaleza",
                 "cole_jornada", "cole_bilingue", "cole_caracter", "cole_calendario"]

num_feats = ["punt_matematicas", "punt_lectura_critica", "punt_c_naturales",
             "punt_sociales_ciudadanas", "punt_ingles", "punt_global", "edad"]


def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop(TARGET)
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


def submuestrear_curva(arr_x, arr_y, n=200):
    if len(arr_x) <= n:
        return arr_x.tolist(), arr_y.tolist()
    idx = np.linspace(0, len(arr_x) - 1, n).astype(int)
    return arr_x[idx].tolist(), arr_y[idx].tolist()


def main():
    print("Leyendo datos...")
    df = pd.read_csv(DATA_PATH)
    print(f"  CSV: {df.shape[0]:,} filas x {df.shape[1]} columnas")

    all_feats = cat_str_feats + num_feats
    df_model  = df[all_feats + [TARGET]].dropna().copy()
    df_model[TARGET] = (df_model[TARGET] == "URBANO").astype(int)

    print("Reproduciendo split train/test...")
    train = df_model.sample(frac=0.8, random_state=SEED)
    test  = df_model.drop(train.index)
    val   = train.sample(frac=0.2, random_state=SEED)
    train = train.drop(val.index)
    print(f"  Train: {len(train):,}   Val: {len(val):,}   Test: {len(test):,}")

    test_ds = dataframe_to_dataset(test).batch(32)

    print("Cargando modelo...")
    model = keras.models.load_model(MODEL_DIR / "modelo_urbano_rural.keras")

    print("Calculando predicciones...")
    y_pred_prob = model.predict(test_ds, verbose=0).flatten()
    y_pred      = (y_pred_prob >= 0.5).astype(int)
    y_true      = np.concatenate([y for x, y in test_ds], axis=0)

    print("Guardando metricas_test.json...")
    metricas = {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "auc":       round(float(roc_auc_score(y_true, y_pred_prob)), 4),
        "recall":    round(float(recall_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "f1":        round(float(f1_score(y_true, y_pred)), 4),
        "n_test":    int(len(y_true))
    }
    with open(MODEL_DIR / "metricas_test.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)
    print(f"  accuracy={metricas['accuracy']}  auc={metricas['auc']}  f1={metricas['f1']}")

    print("Guardando matriz_confusion.json...")
    cm = confusion_matrix(y_true, y_pred)
    cm_dict = {
        "labels": ["Rural", "Urbano"],
        "matrix": cm.tolist(),
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        "threshold": 0.5
    }
    with open(MODEL_DIR / "matriz_confusion.json", "w", encoding="utf-8") as f:
        json.dump(cm_dict, f, indent=2, ensure_ascii=False)

    print("Guardando curvas_roc.json...")
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc     = auc(fpr, tpr)
    fpr_s, tpr_s = submuestrear_curva(fpr, tpr)
    roc_dict = {"fpr": fpr_s, "tpr": tpr_s, "auc": round(float(roc_auc), 4)}
    with open(MODEL_DIR / "curvas_roc.json", "w", encoding="utf-8") as f:
        json.dump(roc_dict, f, ensure_ascii=False)

    print("Guardando comparacion_modelos.csv...")
    comparacion = pd.DataFrame({
        "experimento":  ["E1_Solo_Socioeconomicas", "E2_Solo_Puntajes", "E3_Todas_Variables"],
        "accuracy":     [0.8397, 0.8224, 0.8412],
        "auc":          [0.8443, 0.6631, 0.8506],
        "recall":       [0.9562, 1.0000, 0.9642],
        "precision":    [0.8635, 0.8224, 0.8598]
    })
    comparacion.to_csv(MODEL_DIR / "comparacion_modelos.csv", index=False, encoding="utf-8")

    print("Guardando metadata_luis.json...")
    metadata = {
        "modelo": "Red Neuronal Densa - Clasificacion Binaria",
        "pregunta_negocio": "Es posible predecir si un estudiante proviene de una zona rural o urbana a partir de sus resultados academicos y perfil familiar?",
        "target": TARGET,
        "clases": {"0": "RURAL", "1": "URBANO"},
        "arquitectura": "64-32-16 + Dropout",
        "optimizer": "adam",
        "loss": "binary_crossentropy",
        "epochs": 50,
        "batch_size": 32,
        "cat_str_feats": cat_str_feats,
        "num_feats": num_feats,
        "particion": {
            "n_train": int(len(train)),
            "n_val":   int(len(val)),
            "n_test":  int(len(test)),
            "random_state": SEED
        },
        "metricas_test": metricas
    }
    with open(MODEL_DIR / "metadata_luis.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\nListo. Artefactos en models/luis/")
    for archivo in ["metricas_test.json", "matriz_confusion.json",
                    "curvas_roc.json", "comparacion_modelos.csv", "metadata_luis.json"]:
        path = MODEL_DIR / archivo
        if path.exists():
            print(f"  {archivo:30s}  {path.stat().st_size/1024:>8.1f} KB")


if __name__ == "__main__":
    main()
