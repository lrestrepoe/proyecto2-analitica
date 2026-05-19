import pandas as pd
from pathlib import Path
import unicodedata


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

CARPETA_DATOS = Path(__file__).resolve().parent
NOMBRE_ARCHIVO = "DatosSaber11_Bolivar.csv"

PERIODO_MINIMO = 20141

COL_ID = "estu_consecutivo"
COL_FECHA_NAC = "estu_fechanacimiento"
COL_PERIODO = "periodo"
COL_PUNT_GLOBAL = "punt_global"


# ============================================================
# 2. FUNCIONES AUXILIARES
# ============================================================

def normalizar_texto(x):
    if pd.isna(x):
        return pd.NA
    x = str(x).strip().upper()
    x = unicodedata.normalize("NFKD", x)
    x = "".join(c for c in x if not unicodedata.combining(c))
    return x


def periodo_a_fecha_aprox(periodo):
    if pd.isna(periodo):
        return pd.NaT

    p = str(periodo).strip()

    if len(p) < 5:
        return pd.NaT

    try:
        anio = int(p[:4])
        trimestre = int(p[-1])
    except:
        return pd.NaT

    if trimestre not in [1, 2, 3, 4]:
        return pd.NaT

    mes_por_trimestre = {
        1: 2,
        2: 5,
        3: 8,
        4: 11
    }

    return pd.Timestamp(
        year=anio,
        month=mes_por_trimestre[trimestre],
        day=15
    )


def imprimir_reporte_faltantes(df, nombre):
    rep = pd.DataFrame({
        "columna": df.columns,
        "nulos": [df[c].isna().sum() for c in df.columns],
        "porcentaje_nulos": [round(df[c].isna().mean() * 100, 2) for c in df.columns],
    }).sort_values("porcentaje_nulos", ascending=False)

    print(f"\nREPORTE DE FALTANTES -> {nombre}")
    print(rep.to_string(index=False))

    return rep


def verificar_sin_nulos_y_conteo(df, nombre):
    n = len(df)
    conteos = df.count()

    min_c = int(conteos.min()) if len(conteos) else 0
    max_c = int(conteos.max()) if len(conteos) else 0
    nulos_total = int(df.isna().sum().sum())

    print(f"\nVERIFICACIÓN -> {nombre}")
    print(f"Filas: {n:,}")
    print(f"Columnas: {df.shape[1]:,}")
    print(f"Nulos totales: {nulos_total:,}")
    print(f"Non-null min: {min_c:,}")
    print(f"Non-null max: {max_c:,}")

    if nulos_total == 0 and min_c == n and max_c == n:
        print("OK: el dataframe no tiene valores faltantes.")
    else:
        print("ALERTA: todavía existen valores faltantes.")


def reordenar_columnas(df_):
    orden_prioritario = [
        "estu_consecutivo",
        "periodo",
        "estu_fechanacimiento",
        "edad",
        "estu_genero",

        "cole_area_ubicacion",
        "cole_mcpio_ubicacion",
        "cole_calendario",
        "cole_bilingue",
        "cole_naturaleza",
        "cole_genero",
        "cole_caracter",

        "fami_estratovivienda",
        "fami_tienecomputador",
        "fami_tieneinternet",
        "fami_educacionmadre",
        "fami_educacionpadre",

        "punt_ingles",
        "punt_matematicas",
        "punt_lectura_critica",
        "punt_c_naturales",
        "punt_sociales_ciudadanas",
        "punt_global",
    ]

    cols_base = [c for c in orden_prioritario if c in df_.columns]
    cols_restantes = [c for c in df_.columns if c not in cols_base]

    return df_[cols_base + cols_restantes]


# ============================================================
# 3. LECTURA DEL ARCHIVO
# ============================================================

ruta = Path(CARPETA_DATOS)
archivo = ruta / NOMBRE_ARCHIVO

if not archivo.exists() and not NOMBRE_ARCHIVO.lower().endswith(".csv"):
    archivo = ruta / f"{NOMBRE_ARCHIVO}.csv"

if not archivo.exists():
    raise FileNotFoundError(f"No encontré el archivo en: {archivo}")

print(f"Leyendo archivo: {archivo}")

try:
    df_raw = pd.read_csv(
        archivo,
        dtype=str,
        low_memory=False,
        encoding="utf-8"
    )
except UnicodeDecodeError:
    df_raw = pd.read_csv(
        archivo,
        dtype=str,
        low_memory=False,
        encoding="latin1"
    )

df_raw.columns = df_raw.columns.str.strip().str.lower()

print(f"Filas totales leídas: {len(df_raw):,}")
print(f"Columnas originales del CSV: {df_raw.shape[1]:,}")


# ============================================================
# 4. VALIDACIÓN DE COLUMNAS BÁSICAS
# ============================================================

columnas_necesarias = [
    COL_PERIODO,
    COL_PUNT_GLOBAL
]

faltantes_necesarias = [
    c for c in columnas_necesarias if c not in df_raw.columns
]

if faltantes_necesarias:
    raise ValueError(
        "Faltan columnas necesarias para el procesamiento: "
        + ", ".join(faltantes_necesarias)
    )


# ============================================================
# 5. SE TRABAJA CON TODAS LAS COLUMNAS DEL CSV
# ============================================================

df = df_raw.copy()

columnas_originales_csv = list(df.columns)


# ============================================================
# 6. LIMPIEZA GENERAL DE VACÍOS
# ============================================================

for c in df.columns:
    if df[c].dtype == "object":
        df[c] = df[c].str.strip()

df = df.replace({
    "": pd.NA,
    " ": pd.NA,
    "NA": pd.NA,
    "N/A": pd.NA,
    "NULL": pd.NA,
    "NONE": pd.NA,
    "NAN": pd.NA
})


# ============================================================
# 7. NORMALIZACIÓN DE VARIABLES CATEGÓRICAS PRINCIPALES
# ============================================================

categ_cols = set([
    "cole_bilingue",
    "cole_area_ubicacion",
    "cole_naturaleza",
    "cole_calendario",
    "cole_mcpio_ubicacion",
    "cole_genero",
    "cole_caracter",
    "estu_genero",
    "fami_estratovivienda",
    "fami_tienecomputador",
    "fami_tieneinternet",
    "fami_educacionmadre",
    "fami_educacionpadre",
    "desemp_ingles",
])

for c in df.columns:
    if c in categ_cols:
        df[c] = df[c].apply(normalizar_texto)

if "cole_bilingue" in df.columns:
    df["cole_bilingue"] = df["cole_bilingue"].replace({
        "SI": "S",
        "NO": "N"
    })


# ============================================================
# 8. CONVERSIÓN DE VARIABLES CLAVE
# ============================================================

df[COL_PUNT_GLOBAL] = pd.to_numeric(
    df[COL_PUNT_GLOBAL],
    errors="coerce"
)

df["periodo_int"] = pd.to_numeric(
    df[COL_PERIODO],
    errors="coerce"
)


# ============================================================
# 9. REPORTE DE punt_global POR PERIODO
# ============================================================

resumen_periodo = (
    df.groupby(COL_PERIODO)[COL_PUNT_GLOBAL]
    .agg(
        total="size",
        nulos=lambda s: s.isna().sum()
    )
    .reset_index()
)

resumen_periodo["pct_nulos"] = (
    resumen_periodo["nulos"] / resumen_periodo["total"] * 100
).round(2)

resumen_periodo = resumen_periodo.sort_values(
    "pct_nulos",
    ascending=False
)

print("\nPERIODOS DONDE punt_global ESTÁ VACÍO")
print(resumen_periodo.head(20).to_string(index=False))


# ============================================================
# 10. FILTRO DE PERIODO
# ============================================================

antes = len(df)

df = df[
    df["periodo_int"].notna()
    & (df["periodo_int"] >= PERIODO_MINIMO)
].copy()

print(
    f"\nFilas después de filtrar periodo >= {PERIODO_MINIMO}: "
    f"{len(df):,} (antes: {antes:,})"
)


# ============================================================
# 11. FILTRO DE punt_global
# ============================================================

antes = len(df)

df = df.dropna(subset=[COL_PUNT_GLOBAL]).copy()

print(
    f"Filas después de eliminar punt_global vacío: "
    f"{len(df):,} (antes: {antes:,})"
)

antes = len(df)

df = df[
    (df[COL_PUNT_GLOBAL] >= 0)
    & (df[COL_PUNT_GLOBAL] <= 500)
].copy()

print(
    f"Filas después de eliminar punt_global aberrante: "
    f"{len(df):,} (antes: {antes:,})"
)


# ============================================================
# 12. VALIDACIÓN DE PRUEBAS 0-100
# ============================================================

pruebas_0_100 = [
    "punt_ingles",
    "punt_matematicas",
    "punt_lectura_critica",
    "punt_c_naturales",
    "punt_sociales_ciudadanas",
]

for col in pruebas_0_100:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

        antes = len(df)

        df = df[
            df[col].isna()
            | ((df[col] >= 0) & (df[col] <= 100))
        ].copy()

        print(
            f"Filas después de validar {col} entre 0 y 100: "
            f"{len(df):,} (antes: {antes:,})"
        )


# ============================================================
# 13. DUPLICADOS
# ============================================================

dup_exact = df.duplicated().sum()
print(f"\nDuplicados exactos, considerando todas las columnas: {dup_exact:,}")

df = df.drop_duplicates(keep="first").copy()

if COL_ID in df.columns:
    filas_repetidas_id = df[COL_ID].duplicated(keep=False).sum()

    print(
        f"Filas que pertenecen a IDs repetidos "
        f"(mismo {COL_ID}) tras quitar exactos: {filas_repetidas_id:,}"
    )

    if filas_repetidas_id > 0:
        df = df.sort_values(
            by=[COL_ID, COL_PUNT_GLOBAL],
            ascending=[True, False]
        ).copy()

        antes = len(df)

        df = df.drop_duplicates(
            subset=[COL_ID],
            keep="first"
        ).copy()

        print(
            f"Filas después de quedarnos con el mayor punt_global "
            f"por {COL_ID}: {len(df):,} (antes: {antes:,})"
        )


# ============================================================
# 14. CÁLCULO DE EDAD
# ============================================================

if COL_FECHA_NAC in df.columns:
    df["fecha_nac_dt"] = pd.to_datetime(
        df[COL_FECHA_NAC],
        errors="coerce",
        dayfirst=True
    )
else:
    df["fecha_nac_dt"] = pd.NaT

df["fecha_examen_aprox"] = df[COL_PERIODO].apply(periodo_a_fecha_aprox)

df["edad"] = (
    df["fecha_examen_aprox"] - df["fecha_nac_dt"]
).dt.days / 365.25

df["edad"] = df["edad"].apply(
    lambda x: int(x) if pd.notna(x) else pd.NA
)


# ============================================================
# 15. REPORTE ANTES DE ELIMINAR FALTANTES EN TODAS LAS COLUMNAS
# ============================================================

imprimir_reporte_faltantes(
    df[columnas_originales_csv + ["edad"]],
    "antes_de_eliminar_faltantes_todas_las_columnas"
)


# ============================================================
# 16. DATAFRAME FINAL ÚNICO
# ============================================================

columnas_finales = columnas_originales_csv + ["edad"]

antes = len(df)

df_final = df[columnas_finales].dropna(
    axis=0,
    how="any"
).copy()

print(
    f"\nFilas después de eliminar TODOS los nulos "
    f"considerando TODAS las columnas del CSV + edad: "
    f"{len(df_final):,} (antes: {antes:,})"
)

df_final = reordenar_columnas(df_final)


# ============================================================
# 17. VERIFICACIÓN FINAL
# ============================================================

verificar_sin_nulos_y_conteo(
    df_final,
    "df_final"
)

print("\nPrimeras filas de df_final:")
print(df_final.head())

print("\nInformación de df_final:")
print(df_final.info())


# ============================================================
# 18. EXPORTAR BASE LIMPIA
# ============================================================

salida = ruta / "DatosSaber11_Bolivar_limpio_todas_columnas.csv"

df_final.to_csv(
    salida,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nArchivo exportado correctamente en:")
print(salida)