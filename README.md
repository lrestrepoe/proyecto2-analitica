# Dashboard Saber 11 Bolívar — Proyecto 2

Dashboard en Streamlit que integra tres modelos sobre la base del ICFES Saber 11
para Bolívar.

## Estructura

```
proyecto2-analitica/
├── app.py                              # Entrada del dashboard
├── pages/
│   ├── 1_Panorama_General.py
│   ├── 2_Puntaje_Global.py             # Juan Felipe
│   ├── 3_Rezago_por_Area.py            # Daniel
│   └── 4_Riesgo_Academico.py           # Luis (placeholder)
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── model_loader.py
│   ├── plots.py
│   ├── prediction.py
│   └── ui.py
├── scripts/
│   ├── prepare_artifacts_daniel.py
│   └── prepare_artifacts_juan_felipe.py
├── data/
│   └── DatosSaber11_Bolivar_limpio_todas_columnas.csv
├── models/
│   ├── juan_felipe/
│   └── daniel/
└── requirements.txt
```

## Ejecución local (Windows + VS Code)

1. Abrir terminal en la raíz del proyecto.
2. Crear entorno virtual (una sola vez):
   ```
   py -m venv .venv
   .venv\Scripts\activate
   ```
3. Instalar dependencias:
   ```
   pip install -r requirements.txt
   ```
4. Generar artefactos auxiliares (una sola vez, ya están si los scripts corrieron):
   ```
   py scripts\prepare_artifacts_daniel.py
   py scripts\prepare_artifacts_juan_felipe.py
   ```
5. Levantar el dashboard:
   ```
   py -m streamlit run app.py
   ```
6. Streamlit abre el navegador en `http://localhost:8501`.

## Compartir en red local

Streamlit imprime un `Network URL` (ej. `http://192.168.x.x:8501`) que otros
miembros del equipo pueden abrir si están en la misma red WiFi.

## Notas de diseño

- El dashboard nunca entrena ni recalcula. Solo lee artefactos serializados.
- Datos cacheados con `@st.cache_data`, modelos con `@st.cache_resource`.
- Cada página carga lazy solo sus propios artefactos.
- Preparado para despliegue en EC2 pequeña (TensorFlow-CPU, sin GPU).
