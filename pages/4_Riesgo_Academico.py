"""Página de Luis — clasificación de riesgo académico (placeholder)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.config import MODELS_DIR
from src.ui import encabezado_pagina, separador

st.set_page_config(page_title="Riesgo Académico", layout="wide")

encabezado_pagina(
    "Riesgo Académico",
    "Modelo de clasificación complementario — Luis",
    pregunta="Pregunta de negocio pendiente de definir según el enfoque "
             "final del modelo.",
)

st.info(
    "**Modelo de Luis pendiente de integración.** Esta página queda preparada "
    "con la arquitectura del dashboard para incorporar los artefactos cuando "
    "estén disponibles."
)

separador()

st.markdown("### Secciones previstas")

c1, c2 = st.columns(2)
with c1:
    st.markdown(
        """
        **Análisis exploratorio**
        - Distribución de la variable objetivo.
        - Perfil de estudiantes en riesgo.
        - Comparación entre categorías.
        """
    )
    st.markdown(
        """
        **Métricas del modelo**
        - Accuracy, Precision, Recall, F1, AUC.
        - Comparación de modelos (si se entrena más de uno).
        """
    )

with c2:
    st.markdown(
        """
        **Validación gráfica**
        - Curva ROC.
        - Curva Precision-Recall.
        - Matriz de confusión.
        - Importancia de variables.
        """
    )
    st.markdown(
        """
        **Simulador**
        - Predicción individual a partir de perfil ingresado.
        - Recomendación interpretativa según resultado.
        """
    )

separador()

st.markdown("### Artefactos esperados")

archivos_esperados = [
    "modelo_luis.keras",
    "preprocessor_luis.joblib",
    "metricas_test.json",
    "curva_roc.json",
    "curva_pr.json",
    "matriz_confusion.json",
    "importancia_permutation.csv",
]

luis_dir = MODELS_DIR / "luis"
luis_dir.mkdir(parents=True, exist_ok=True)

estado = []
for archivo in archivos_esperados:
    path = luis_dir / archivo
    estado.append({"Archivo": archivo, "Estado": "✅ Encontrado" if path.exists() else "⏳ Pendiente"})

import pandas as pd
st.dataframe(pd.DataFrame(estado), use_container_width=True, hide_index=True)

st.caption(
    "Cuando los artefactos estén en `models/luis/`, esta página se actualizará "
    "automáticamente para mostrar EDA, métricas, curvas, importancia y simulador, "
    "siguiendo la misma estructura modular del resto del dashboard."
)
