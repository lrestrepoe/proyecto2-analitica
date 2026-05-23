"""Configuración central del dashboard."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "DatosSaber11_Bolivar_limpio_todas_columnas.csv"
MODELS_DIR = ROOT / "models"
MODELS_JF = MODELS_DIR / "juan_felipe"
MODELS_DANI = MODELS_DIR / "daniel"
MODELS_LUIS = MODELS_DIR / "luis"

COLOR_PRIMARY = "#1B3A6B"
COLOR_SECONDARY = "#2E6DB4"
COLOR_ACCENT = "#C0392B"
COLOR_NEUTRAL = "#D9DEE4"
COLOR_DARK = "#2C3E50"

PALETTE_CATEGORICAL = [
    "#1B3A6B", "#2E6DB4", "#C0392B", "#5DADE2",
    "#E67E22", "#2C3E50", "#16A085", "#8E44AD",
]
PALETTE_SEQUENTIAL = ["#D9DEE4", "#A9C0DB", "#7AA2C8", "#4A85B5", "#2E6DB4", "#1B3A6B"]

PLOTLY_TEMPLATE = "simple_white"

AREAS = {
    "Matemáticas": "punt_matematicas",
    "Lectura Crítica": "punt_lectura_critica",
    "Sociales y Ciudadanas": "punt_sociales_ciudadanas",
    "Ciencias Naturales": "punt_c_naturales",
    "Inglés": "punt_ingles",
}

UMBRAL_RIESGO_ALTO = 0.60
UMBRAL_RIESGO_MEDIO = 0.35
