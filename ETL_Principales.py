"""
====================================================================
ETL Unificado - Datos Fiscales y Macroeconomicos de Mexico
Autor: Jimenez Flores Luis Arturo
Ultima modificacion: 17 de Diciembre del 2025
Descripcion: Procesamiento integral de datos trimestrales de fuentes
gubernamentales. Genera datasets consolidados en USD corrientes y
USD constantes 2018.
====================================================================
"""

import json
import re
import unicodedata
import warnings
from pathlib import Path
from datetime import datetime
from functools import wraps

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURACION DE RUTAS
# ============================================================================

BASE_DIR = Path(__file__).parent.resolve()

# Directorios de salida
OUTPUT_ROOT = BASE_DIR / "Datos_Resultado"
VALIDACIONES_DIR = OUTPUT_ROOT / "Validaciones"
DATASET_FINAL_DIR = OUTPUT_ROOT / "DatasetFinal"
INDICADORES_DIR = DATASET_FINAL_DIR / "Indicadores"

# Rutas de entrada
RUTAS_ENTRADA = {
    "deuda_hist": BASE_DIR / "Datos_Origen/DatosPrincipales/DeudaPublica/Historica/deuda_publica_hist.csv",
    "deuda_act": BASE_DIR / "Datos_Origen/DatosPrincipales/DeudaPublica/Actual/deuda_publica.csv",
    "rfsp_hist": BASE_DIR / "Datos_Origen/DatosPrincipales/DeficitFiscal_RFSP/Trimestres_Anterior/rfsp_metodologia_anterior.csv",
    "rfsp_act": BASE_DIR / "Datos_Origen/DatosPrincipales/DeficitFiscal_RFSP/Mensual_Actual/rfsp.csv",
    "balance_hist": BASE_DIR / "Datos_Origen/DatosPrincipales/BalancePublico_IngresoGastoFinanciamiento/Historica/ingreso_gasto_finan_hist.csv",
    "balance_act": BASE_DIR / "Datos_Origen/DatosPrincipales/BalancePublico_IngresoGastoFinanciamiento/Actual/ingreso_gasto_finan.csv",
    "pib_nominal": BASE_DIR / "Datos_Origen/DatosPrincipales/PIB/PIBT_3.xlsx",
    "pib_real": BASE_DIR / "Datos_Origen/DatosPrincipales/PIB/PIBT_2.xlsx",
    "inflacion": BASE_DIR / "Datos_Origen/DatosPrincipales/Inflacion/inflacion.csv",
    "tipo_cambio": BASE_DIR / "Datos_Origen/DatosPrincipales/TipoCambio/tipo_cambio.xlsx",
    "balanza_pagos": BASE_DIR / "Datos_Origen/DatosPrincipales/BalanzaPagos/BalanzaPagos.xlsx",
    "tasas_mx": BASE_DIR / "Datos_Origen/DatosPrincipales/TasasInteres/Mexico/Tasas_Interes_Mexico.xlsx",
    "tasas_us": BASE_DIR / "Datos_Origen/DatosPrincipales/TasasInteres/USA/FEDFUNDS.csv",
    "cpi": BASE_DIR / "Datos_Origen/DatosPrincipales/Deflactor/CPIAUCSL.csv",
}

# Rutas de salida
RUTAS_SALIDA = {
    "deuda_externa": OUTPUT_ROOT / "mexico_deuda_externa_core_mUSD.csv",
    "balance_wide": OUTPUT_ROOT / "mexico_balance_publico_Trimestral_SHCP_Original.csv",
    "balance_agg": OUTPUT_ROOT / "balance_publico_agregados.csv",
    "rfsp": OUTPUT_ROOT / "rfsp_deficit_core.csv",
    "pib": OUTPUT_ROOT / "PIB_procesado.csv",
    "inflacion": OUTPUT_ROOT / "Inflacion_Trimestral.csv",
    "tipo_cambio": OUTPUT_ROOT / "TipoCambio_Trimestral.csv",
    "balanza_pagos": OUTPUT_ROOT / "BalanzaPagos_Trimestral.csv",
    "tasas": OUTPUT_ROOT / "TasaInteres_Trimestral.csv",
    "final_corrientes": DATASET_FINAL_DIR / "DatasetFinal_USD_Corrientes.csv",
    "final_2018": DATASET_FINAL_DIR / "DatasetFinal_USD_2018.csv",
    "final_corrientes_limpio": DATASET_FINAL_DIR / "DatasetFinal_USD_Corrientes_limpio.csv",
    "final_2018_limpio": DATASET_FINAL_DIR / "DatasetFinal_USD_2018_limpio.csv",
    "final_corrientes_reducido": DATASET_FINAL_DIR / "DatasetFinal_USD_Corrientes_reducido.csv",
    "final_2018_reducido": DATASET_FINAL_DIR / "DatasetFinal_USD_2018_reducido.csv",
}


# ============================================================================
# CONSTANTES Y MAPEOS
# ============================================================================

MESES_TRIM_HIST = {"Marzo": 1, "Junio": 2, "Septiembre": 3, "Diciembre": 4}
MESES_TRIM_ESPECIALES = {"Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4}
MESES_TRIM_12 = {
    "Enero": 1, "Febrero": 1, "Marzo": 1, "Abril": 2, "Mayo": 2, "Junio": 2,
    "Julio": 3, "Agosto": 3, "Septiembre": 3, "Octubre": 4, "Noviembre": 4, "Diciembre": 4
}
ORDEN_MESES = {m: i for i, m in enumerate(
    ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
     "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"], 1)}

SERIE_KEYS = ["CLAVE_DE_CONCEPTO", "NOMBRE", "SUBTEMA", "SECTOR", "AMBITO",
              "TIPO_DE_INFORMACION", "BASE_DE_REGISTRO", "UNIDAD_DE_MEDIDA", "FRECUENCIA", "DIFUSION", "CICLO"]
SERIE_KEYS_RFSP = ["CLAVE_DE_CONCEPTO", "NOMBRE", "TEMA", "SUBTEMA", "SECTOR", "AMBITO",
                   "TIPO_DE_INFORMACION", "BASE_DE_REGISTRO", "UNIDAD_DE_MEDIDA", "FRECUENCIA", "CICLO"]

TARGETS_DEUDA_EXTERNA = {
    "deuda_total_economia_musd": ["xem10_"], "deuda_bruta_publica_musd": ["xeb20_"],
    "activos_fx_publico_musd": ["xeb10_"], "posicion_neta_publico_musd": ["xeb00_"],
    "deuda_publica_lp_musd": ["xeb3010_"], "deuda_publica_cp_musd": ["xeb3020_"],
    "deuda_gob_federal_musd": ["xeb4010_"], "deuda_organismos_empresas_musd": ["xeb4020_"],
    "deuda_banca_desarrollo_musd": ["xeb4030_"], "endeudamiento_neto_total_musd": ["xebc20_"],
    "endeudamiento_neto_lp_musd": ["xebc3010_"], "endeudamiento_neto_cp_musd": ["xebc3020_"],
    "endeudamiento_neto_gf_musd": ["xebc4010_"], "endeudamiento_neto_oye_musd": ["xebc4020_"],
    "endeudamiento_neto_bd_musd": ["xebc4030_"], "fuente_bancario_musd": ["xec20_"],
    "fuente_comercio_ext_musd": ["xec40_"], "fuente_capitales_musd": ["xec50_"],
    "fuente_ofis_musd": ["xec60_"], "fuente_pidiregas_musd": ["xec80_"],
    "acreedor_eeuu_musd": ["xeh10_"], "acreedor_ofis_musd": ["xeh20_"],
    "acreedor_japon_musd": ["xeh30_"], "acreedor_otros_musd": ["xeh90_"],
    "moneda_usd_musd": ["xei10_"], "moneda_canasta_musd": ["xei20_"],
}


# ============================================================================
# FUNCIONES AUXILIARES - Carga de archivos y manejo de errores
# ============================================================================

def inicializar_directorios():
    """Crea todos los directorios de salida necesarios."""
    for d in [OUTPUT_ROOT, VALIDACIONES_DIR, DATASET_FINAL_DIR, INDICADORES_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"+ Directorios creados en: {OUTPUT_ROOT}")


def con_manejo_errores(nombre_proceso):
    """Decorador para manejo uniforme de errores en procesadores."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"x Error en {nombre_proceso}: {e}")
                return pd.DataFrame()
        return wrapper
    return decorator


def cargar_csv_shcp(path_hist, path_act):
    """Carga y concatena archivos CSV historico y actual de SHCP."""
    if not path_hist.exists() or not path_act.exists():
        return None
    hist = pd.read_csv(path_hist, encoding="latin1")
    act = pd.read_csv(path_act, encoding="utf-8")
    return pd.concat([hist, act], ignore_index=True)


def cargar_excel(path, **kwargs):
    """Carga archivo Excel con manejo de errores."""
    if not path.exists():
        return None
    return pd.read_excel(path, **kwargs)


def guardar_csv(df, path, msg=None):
    """Guarda DataFrame como CSV con mensaje de confirmacion."""
    df.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")
    if msg:
        print(f"+ {msg}: {path.name} | {df.shape[0]} filas x {df.shape[1]} cols")


def exportar_validacion(df, filename):
    """Guarda validaciones en carpeta Validaciones."""
    path = VALIDACIONES_DIR / filename
    df.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def validacion_basica(df, nombre, periodo_col="periodo", cols=None):
    """Genera resumen de calidad para un dataset."""
    if df is None or df.empty:
        return pd.DataFrame([{"check": "vacio", "valor": 0, "nota": f"{nombre}: sin datos"}])

    d = df.copy()
    d[periodo_col] = pd.to_datetime(d[periodo_col], errors='coerce')
    d = d.dropna(subset=[periodo_col]).sort_values(periodo_col)

    rows = [
        {"check": "filas", "valor": len(d), "nota": nombre},
        {"check": "columnas", "valor": d.shape[1], "nota": nombre},
        {"check": "min_periodo", "valor": d[periodo_col].min().strftime('%Y-%m-%d') if len(d) else '', "nota": nombre},
        {"check": "max_periodo", "valor": d[periodo_col].max().strftime('%Y-%m-%d') if len(d) else '', "nota": nombre},
    ]

    for c in (cols or [c for c in d.columns if c != periodo_col]):
        if c in d.columns:
            pct = pd.to_numeric(d[c], errors='coerce').notna().mean() * 100
            rows.append({"check": "completitud", "valor": round(pct, 2), "nota": f"{nombre}:{c}"})

    return pd.DataFrame(rows)


# ============================================================================
# FUNCIONES AUXILIARES - Normalizacion y conversion
# ============================================================================

def normalizar_texto(s):
    """Normaliza texto: elimina acentos y convierte a minusculas."""
    s = "" if pd.isna(s) else str(s)
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.category(c).startswith("M")).lower().strip()


def safe_col(s):
    """Convierte string a nombre de columna seguro."""
    s = str(s).replace(" ", "_")
    for a, b in [("a", "a"), ("e", "e"), ("i", "i"), ("o", "o"), ("u", "u"), ("n", "n")]:
        s = s.replace(a, b)
    return re.sub(r"_+", "_", re.sub(r"[^0-9A-Za-z_]+", "_", s)).strip("_")


def mes_norm(x):
    """Normaliza nombres de meses."""
    x = normalizar_texto(x or "")
    x = re.sub(r"[^\w]+", "", x)
    MAP = {"enero": "Enero", "febrero": "Febrero", "marzo": "Marzo", "abril": "Abril",
           "mayo": "Mayo", "junio": "Junio", "julio": "Julio", "agosto": "Agosto",
           "septiembre": "Septiembre", "setiembre": "Septiembre", "octubre": "Octubre",
           "noviembre": "Noviembre", "diciembre": "Diciembre",
           "ene": "Enero", "feb": "Febrero", "mar": "Marzo", "abr": "Abril",
           "may": "Mayo", "jun": "Junio", "jul": "Julio", "ago": "Agosto",
           "sep": "Septiembre", "oct": "Octubre", "nov": "Noviembre", "dic": "Diciembre"}
    return MAP.get(x)


def periodo_trimestre(year, quarter):
    """Retorna string YYYY-MM-DD del primer dia del trimestre."""
    return f"{int(year):04d}-{(int(quarter) - 1) * 3 + 1:02d}-01"


def build_periodo(df):
    """Construye columna periodo trimestral desde CICLO y TRIMESTRE."""
    y = pd.to_numeric(df["CICLO"], errors="coerce").astype("Int64")
    q = pd.to_numeric(df["TRIMESTRE"], errors="coerce").astype("Int64")
    m = (q * 3 - 2).astype("Int64")
    return pd.to_datetime(y.astype(str).str.zfill(4) + "-" + m.astype(str).str.zfill(2) + "-01", errors="coerce")


def convertir_usd_miles_a_musd(monto, unidad):
    """Convierte miles de USD a millones de USD."""
    u = normalizar_texto(unidad)
    val = pd.to_numeric(monto, errors="coerce")
    if pd.isna(val):
        return np.nan
    if ("mile" in u) and (("dolar" in u) or ("usd" in u)):
        return val / 1000.0
    return np.nan


def convertir_a_millones_mxn(monto, unidad):
    """Convierte montos a millones de MXN."""
    u = normalizar_texto(unidad)
    val = pd.to_numeric(monto, errors="coerce")
    if pd.isna(val):
        return np.nan
    if ("mile" in u) and ("peso" in u):
        return val / 1000.0
    if ("millon" in u) and ("peso" in u):
        return val
    return np.nan


def excel_to_datetime(x):
    """Convierte serial de Excel o texto a datetime."""
    if pd.isna(x):
        return pd.NaT
    if isinstance(x, (pd.Timestamp, datetime)):
        return pd.to_datetime(x)
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        try:
            return pd.Timestamp("1899-12-30") + pd.to_timedelta(int(x), unit="D")
        except:
            return pd.NaT
    return pd.to_datetime(x, errors="coerce")


# ============================================================================
# FUNCIONES AUXILIARES - Trimestrizacion
# ============================================================================

def map_trim_por_serie(grp):
    """Mapea meses a trimestres segun el patron de la serie."""
    uniq = set(grp["MES"].dropna().unique().tolist())
    if uniq and uniq.issubset(set(MESES_TRIM_HIST.keys())):
        return grp["MES"].map(MESES_TRIM_HIST)
    if uniq and uniq.issubset(set(MESES_TRIM_ESPECIALES.keys())):
        return grp["MES"].map(MESES_TRIM_ESPECIALES)
    return grp["MES"].map(MESES_TRIM_12)


def agregar_trimestral(grp, monto_col="MONTO", tipo_col="TIPO_DE_INFORMACION", orden_col="orden_mes"):
    """Agrega observaciones mensuales a nivel trimestral (para Deuda Externa).

    Reglas:
        - Saldo: ultimo dato del trimestre
        - Flujo (exacto): suma del trimestre
        - Otro: promedio simple
    """
    tipo = normalizar_texto(grp[tipo_col].iloc[0]) if (tipo_col in grp and len(grp) > 0) else ""
    if tipo == "saldo":
        g = grp.assign(_ord=grp[orden_col]).sort_values("_ord")
        val = pd.to_numeric(g[monto_col], errors="coerce").iloc[-1]
    elif tipo == "flujo":
        val = pd.to_numeric(grp[monto_col], errors="coerce").sum(min_count=1)
    else:
        val = pd.to_numeric(grp[monto_col], errors="coerce").mean()
    return pd.Series({monto_col: val})


def agregar_trimestral_rfsp(grp, monto_col="MONTO", tipo_col="TIPO_DE_INFORMACION", orden_col="orden_mes"):
    """Agrega observaciones mensuales a nivel trimestral (para RFSP).

    Reglas RFSP:
        - Flujo (startswith) o sin tipo: suma del trimestre (default)
        - Cualquier otro: ultimo dato del trimestre
    """
    tipo = normalizar_texto(grp[tipo_col].iloc[0]) if (tipo_col in grp and len(grp) > 0) else ""
    if (not tipo) or tipo.startswith("flujo"):
        val = pd.to_numeric(grp[monto_col], errors="coerce").sum(min_count=1)
    else:
        g = grp.assign(_ord=grp[orden_col]).sort_values("_ord")
        val = pd.to_numeric(g[monto_col], errors="coerce").iloc[-1]
    return pd.Series({monto_col: val})


def find_col_prefix(columns, prefixes):
    """Busca columna por prefijo de codigo."""
    pats = [re.compile(rf"^{re.escape(p)}(?:_|$)", re.I) for p in prefixes]
    for c in columns:
        for p in pats:
            if p.search(c):
                return c
    return None


# ============================================================================
# SECCION 1: DEUDA EXTERNA - SHCP
# ============================================================================

@con_manejo_errores("Deuda Externa")
def procesar_deuda_externa():
    """Procesa datos de Deuda Externa de SHCP."""
    shcp = cargar_csv_shcp(RUTAS_ENTRADA["deuda_hist"], RUTAS_ENTRADA["deuda_act"])
    if shcp is None:
        print("x No se encontraron CSV de deuda publica")
        return pd.DataFrame()

    mask_ext = shcp["SUBTEMA"].fillna("").apply(lambda x: "deuda externa" in normalizar_texto(x))
    de = shcp.loc[mask_ext].copy()

    de["TRIMESTRE"] = de.groupby(SERIE_KEYS, group_keys=False).apply(map_trim_por_serie)
    de["orden_mes"] = de["MES"].map(ORDEN_MESES)
    de["periodo"] = build_periodo(de)

    cols_group = [c for c in de.columns if c not in ["MES", "orden_mes", "MONTO"]]
    de_q = de.dropna(subset=["TRIMESTRE"]).groupby(cols_group, dropna=False).apply(agregar_trimestral).reset_index()

    de_q["MONTO_mUSD"] = de_q.apply(lambda r: convertir_usd_miles_a_musd(r["MONTO"], r["UNIDAD_DE_MEDIDA"]), axis=1)
    de_q["col_nombre"] = de_q["CLAVE_DE_CONCEPTO"] + "_" + pd.Series(de_q["NOMBRE"], dtype=str).str.replace(r"[^0-9A-Za-z_]+", "_", regex=True)

    wide = de_q.pivot_table(index=["periodo", "CICLO", "TRIMESTRE"], columns="col_nombre",
                            values="MONTO_mUSD", aggfunc="first").reset_index()

    core = wide[["periodo", "CICLO", "TRIMESTRE"]].copy()
    for std_name, prefixes in TARGETS_DEUDA_EXTERNA.items():
        for c in wide.columns:
            if any(normalizar_texto(c).startswith(p) for p in prefixes):
                core[std_name] = pd.to_numeric(wide[c], errors="coerce")
                break

    core = core.sort_values(["periodo", "CICLO", "TRIMESTRE"]).reset_index(drop=True)
    guardar_csv(core, RUTAS_SALIDA["deuda_externa"], "Deuda Externa")
    exportar_validacion(validacion_basica(core, "deuda_externa"), "deuda_externa_validaciones.csv")
    return core


# ============================================================================
# SECCION 2: BALANCE PUBLICO - SHCP
# ============================================================================

@con_manejo_errores("Balance Publico")
def procesar_balance_publico():
    """Procesa Balance Publico de SHCP."""
    df = cargar_csv_shcp(RUTAS_ENTRADA["balance_hist"], RUTAS_ENTRADA["balance_act"])
    if df is None:
        print("x No se encontraron archivos de balance publico")
        return pd.DataFrame()

    df = df.drop_duplicates(subset=["CICLO", "MES", "CLAVE_DE_CONCEPTO", "MONTO"])
    df["MES_norm"] = df["MES"].map(mes_norm)
    df["TRIMESTRE"] = df["MES_norm"].map(MESES_TRIM_12).astype("Int64")
    df["CICLO"] = pd.to_numeric(df["CICLO"], errors="coerce").astype("Int64")
    df["MONTO_mmxn"] = df.apply(lambda r: convertir_a_millones_mxn(r["MONTO"], r["UNIDAD_DE_MEDIDA"]), axis=1)

    core = df[df["CLAVE_DE_CONCEPTO"].isin(["XAB", "XAC", "XAA"])].copy()
    core_q = core.dropna(subset=["CICLO", "TRIMESTRE", "MONTO_mmxn"]).groupby(
        ["CICLO", "TRIMESTRE", "CLAVE_DE_CONCEPTO"], as_index=False).agg(val=("MONTO_mmxn", "sum"))
    core_q["periodo"] = build_periodo(core_q).dt.strftime("%Y-%m-%d")

    wide = core_q.pivot_table(index=["periodo", "CICLO", "TRIMESTRE"], columns="CLAVE_DE_CONCEPTO", values="val").reset_index()
    wide = wide.rename(columns={"XAB": "ingresos_mmxn", "XAC": "gasto_mmxn", "XAA": "balance_mmxn"})
    wide["balance_derivado_mmxn"] = wide["ingresos_mmxn"] - wide["gasto_mmxn"]
    wide["gap_balance_mmxn"] = wide["balance_mmxn"] - wide["balance_derivado_mmxn"]

    guardar_csv(wide.sort_values("periodo"), RUTAS_SALIDA["balance_wide"], "Balance Publico Wide")
    guardar_csv(wide[["periodo", "CICLO", "TRIMESTRE", "ingresos_mmxn", "gasto_mmxn", "balance_mmxn"]],
                RUTAS_SALIDA["balance_agg"], "Balance Publico Agg")
    exportar_validacion(validacion_basica(wide, "balance_publico"), "balance_publico_validaciones.csv")
    return wide


# ============================================================================
# SECCION 3: RFSP (DEFICIT FISCAL) - SHCP
# ============================================================================

@con_manejo_errores("RFSP")
def procesar_rfsp():
    """Procesa RFSP (deficit fiscal) de SHCP."""
    path_hist, path_act = RUTAS_ENTRADA["rfsp_hist"], RUTAS_ENTRADA["rfsp_act"]
    if not path_hist.exists() or not path_act.exists():
        print("x No se encontraron archivos RFSP")
        return pd.DataFrame()

    hist = pd.read_csv(path_hist, encoding="latin1")
    act = pd.read_csv(path_act, encoding="utf-8")

    # Marcar fuente para el traslape 2008-2014
    # Nota: lexicograficamente "actual" < "hist", al ordenar ascending y usar aggfunc="last",
    # "hist" queda al final y prevalece (comportamiento del notebook original)
    hist["_fuente"] = "hist"
    act["_fuente"] = "actual"
    rfsp = pd.concat([hist, act], ignore_index=True, sort=False)

    rfsp["MES"] = rfsp["MES"].map(mes_norm)
    rfsp["TRIMESTRE"] = rfsp.groupby(SERIE_KEYS_RFSP, group_keys=False).apply(map_trim_por_serie)
    rfsp["orden_mes"] = rfsp["MES"].map(ORDEN_MESES)
    rfsp["periodo"] = build_periodo(rfsp)

    cols_group = [c for c in rfsp.columns if c not in ["MES", "orden_mes", "MONTO"]]
    rfsp_q = rfsp.dropna(subset=["TRIMESTRE"]).groupby(cols_group, dropna=False).apply(agregar_trimestral_rfsp).reset_index()
    rfsp_q["col_nombre"] = rfsp_q["CLAVE_DE_CONCEPTO"] + "_" + rfsp_q["NOMBRE"].apply(safe_col)

    # Ordenar para priorizar traslape: lexicograficamente "actual" < "hist",
    # por lo que con ascending=True (default), "hist" queda al final y gana con aggfunc="last"
    rfsp_q = rfsp_q.sort_values(["periodo", "CICLO", "TRIMESTRE", "_fuente"])

    wide = rfsp_q.pivot_table(index=["periodo", "CICLO", "TRIMESTRE"], columns="col_nombre",
                              values="MONTO", aggfunc="last").reset_index()

    COLS = list(wide.columns)
    core = wide[["periodo", "CICLO", "TRIMESTRE"]].copy()

    mappings = [
        ("rfsp_total_miles_mxn", ["RF110"]),
        ("rfsp_sin_ingresos_rec_miles", ["RF120"]),
        ("balance_tradicional_miles", ["RF000001SPFCS", "RF000001SPFC"]),
        ("ingresos_totales_miles", ["RF100000SPFC"]),
        ("gasto_totales_miles", ["RF200000SPFC"]),
        ("incurrimiento_neto_pasivos_miles", ["RF300000SPFC"]),
        ("adq_neta_activos_fin_miles", ["RF400000SPFC"]),
    ]

    for col_name, prefixes in mappings:
        found = find_col_prefix(COLS, prefixes)
        if found:
            core[col_name] = pd.to_numeric(wide[found], errors="coerce")

    guardar_csv(core, RUTAS_SALIDA["rfsp"], "RFSP")
    exportar_validacion(validacion_basica(core, "rfsp"), "rfsp_validaciones.csv")
    return core


# ============================================================================
# SECCION 4: PIB - INEGI
# ============================================================================

@con_manejo_errores("PIB")
def procesar_pib():
    """Procesa datos del PIB del INEGI."""
    def leer_pib(filepath, value_row=7):
        df = cargar_excel(filepath, sheet_name='Tabulado', header=None)
        if df is None:
            return []
        years, quarters, values = df.iloc[4, :], df.iloc[5, :], df.iloc[value_row, :]
        result, current_year = [], None
        for col in range(1, len(df.columns)):
            if pd.notna(years.iloc[col]) and years.iloc[col] != '':
                yv = years.iloc[col]
                current_year = int(''.join(filter(str.isdigit, str(yv)))) if isinstance(yv, str) else int(yv)
            qs, val = quarters.iloc[col], values.iloc[col]
            if pd.notna(qs) and isinstance(qs, str) and qs.startswith('T') and pd.notna(val) and current_year:
                if 2000 <= current_year <= 2030:
                    result.append({'year': current_year, 'quarter': int(qs[1]), 'value': float(val)})
        return result

    d_nom, d_real = leer_pib(RUTAS_ENTRADA["pib_nominal"]), leer_pib(RUTAS_ENTRADA["pib_real"])
    if not d_nom or not d_real:
        print("x Error al leer archivos de PIB")
        return pd.DataFrame()

    df_nom = pd.DataFrame(d_nom).drop_duplicates(subset=['year', 'quarter']).rename(columns={'value': 'PIB_nom_trimestral'})
    df_real = pd.DataFrame(d_real).drop_duplicates(subset=['year', 'quarter']).rename(columns={'value': 'PIB_real_trimestral'})
    df = pd.merge(df_nom, df_real, on=['year', 'quarter'], how='inner')
    df['periodo'] = df.apply(lambda r: periodo_trimestre(r['year'], r['quarter']), axis=1)
    df = df.sort_values('periodo').reset_index(drop=True)[['periodo', 'year', 'quarter', 'PIB_nom_trimestral', 'PIB_real_trimestral']]

    guardar_csv(df, RUTAS_SALIDA["pib"], "PIB")
    exportar_validacion(validacion_basica(df, "pib", cols=['PIB_nom_trimestral', 'PIB_real_trimestral']), "pib_validaciones.csv")
    return df


# ============================================================================
# SECCION 5: INFLACION - INEGI
# ============================================================================

@con_manejo_errores("Inflacion")
def procesar_inflacion():
    """Procesa INPC mensual y construye series trimestrales."""
    MESES = {'Ene': 1, 'Feb': 2, 'Mar': 3, 'Abr': 4, 'May': 5, 'Jun': 6,
             'Jul': 7, 'Ago': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dic': 12,
             'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6,
             'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12}

    def parse_fecha(s):
        if pd.isna(s):
            return None
        s = str(s).strip()
        trozos = s.split()
        if len(trozos) == 2 and trozos[0] in MESES:
            return datetime(int(trozos[1]), MESES[trozos[0]], 1)
        m = re.match(r"^(\d{4})[-/mM](\d{1,2})$", s)
        if m:
            return datetime(int(m.group(1)), int(m.group(2)), 1)
        return None

    path = RUTAS_ENTRADA["inflacion"]
    if not path.exists():
        print("x No se encontro archivo de inflacion")
        return pd.DataFrame()

    # Leer archivo con manejo de líneas irregulares
    enc = "latin1"
    for try_enc in ["cp1252", "latin1", "utf-8"]:
        try:
            with open(path, 'r', encoding=try_enc) as f:
                lines = f.readlines()
            enc = try_enc
            break
        except:
            continue
    else:
        return pd.DataFrame()

    # Buscar inicio de datos (línea que empiece con fecha)
    start = 0
    for i, line in enumerate(lines):
        first_col = line.split(',')[0].strip().strip('"')
        if parse_fecha(first_col):
            start = i
            break

    # Leer solo las 2 primeras columnas desde el inicio de datos
    data_lines = lines[start:]
    rows = []
    for line in data_lines:
        parts = line.split(',')
        if len(parts) >= 2:
            fecha = parts[0].strip().strip('"')
            inpc = parts[1].strip().strip('"')
            rows.append([fecha, inpc])

    df = pd.DataFrame(rows, columns=["fecha", "inpc"])
    df = df[~df["inpc"].isin(["N/E", "NE", "n/e", "", None])].copy()
    df["inpc"] = pd.to_numeric(df["inpc"].str.replace(",", ".").str.replace(" ", ""), errors="coerce")
    df["fecha_dt"] = df["fecha"].apply(parse_fecha)
    df = df.dropna(subset=["fecha_dt", "inpc"]).sort_values("fecha_dt").reset_index(drop=True)
    df["year"], df["quarter"] = df["fecha_dt"].dt.year, df["fecha_dt"].dt.quarter

    q = df.groupby(["year", "quarter"]).agg(
        inpc_q_eoq=("inpc", "last"), inpc_q_avg=("inpc", "mean")).reset_index()
    q["inflacion_yoy_eoq"] = q["inpc_q_eoq"].pct_change(4) * 100
    q["inflacion_yoy_avg"] = q["inpc_q_avg"].pct_change(4) * 100
    q["periodo"] = q.apply(lambda r: periodo_trimestre(r["year"], r["quarter"]), axis=1)

    out = q[q[["inflacion_yoy_eoq", "inflacion_yoy_avg"]].notna().any(axis=1)].copy()
    out = out[["periodo", "year", "quarter", "inpc_q_eoq", "inpc_q_avg", "inflacion_yoy_eoq", "inflacion_yoy_avg"]]

    # Redondeo como notebook original
    out["inpc_q_eoq"] = out["inpc_q_eoq"].round(4)
    out["inpc_q_avg"] = out["inpc_q_avg"].round(4)
    out["inflacion_yoy_eoq"] = out["inflacion_yoy_eoq"].round(2)
    out["inflacion_yoy_avg"] = out["inflacion_yoy_avg"].round(2)

    guardar_csv(out, RUTAS_SALIDA["inflacion"], "Inflacion")
    exportar_validacion(validacion_basica(out, "inflacion"), "inflacion_validaciones.csv")
    return out


# ============================================================================
# SECCION 6: TIPO DE CAMBIO - BANXICO
# ============================================================================

@con_manejo_errores("Tipo de Cambio")
def procesar_tipo_cambio():
    """Procesa tipo de cambio de Banxico."""
    df = cargar_excel(RUTAS_ENTRADA["tipo_cambio"], sheet_name="Hoja1", header=None)
    if df is None:
        print("x No se encontro archivo de tipo de cambio")
        return pd.DataFrame()

    codes = df.iloc[17]
    col_target = None
    for code in ["SF18561", "SF17908"]:
        hit = (codes == code).to_numpy().nonzero()[0]
        if len(hit):
            col_target = int(hit[0])
            break
    if col_target is None:
        col_target = max(range(1, df.shape[1]), key=lambda c: pd.to_numeric(df.iloc[18:, c], errors="coerce").notna().mean())

    tmp = df.loc[18:, [0, col_target]].copy()
    tmp.columns = ["fecha", "tcn"]
    tmp["fecha"] = pd.to_datetime(tmp["fecha"], errors="coerce")
    tmp["tcn"] = pd.to_numeric(tmp["tcn"], errors="coerce").replace(0, np.nan)
    tmp = tmp.dropna(subset=["fecha", "tcn"]).sort_values("fecha")
    tmp["anio"], tmp["tri"] = tmp["fecha"].dt.year, ((tmp["fecha"].dt.month - 1) // 3) + 1

    mean_q = tmp.groupby(["anio", "tri"], as_index=False)["tcn"].mean().rename(columns={"tcn": "tcn_mean"})
    eop_q = tmp.sort_values("fecha").groupby(["anio", "tri"]).tail(1)[["anio", "tri", "tcn"]].rename(columns={"tcn": "tcn_eop"})
    out = mean_q.merge(eop_q, on=["anio", "tri"], how="outer").sort_values(["anio", "tri"])
    out["periodo"] = out.apply(lambda r: periodo_trimestre(r.anio, r.tri), axis=1)
    out = out[["periodo", "tcn_mean", "tcn_eop"]].round(4)

    guardar_csv(out, RUTAS_SALIDA["tipo_cambio"], "Tipo de Cambio")
    exportar_validacion(validacion_basica(out, "tipo_cambio"), "tipo_cambio_validaciones.csv")
    return out


# ============================================================================
# SECCION 7: BALANZA DE PAGOS - BANXICO
# ============================================================================

@con_manejo_errores("Balanza de Pagos")
def procesar_balanza_pagos():
    """Procesa Balanza de Pagos de Banxico."""
    raw = cargar_excel(RUTAS_ENTRADA["balanza_pagos"], sheet_name=0, header=None)
    if raw is None or raw.empty:
        print("x No se encontro archivo de balanza de pagos")
        return pd.DataFrame()

    hdr_row, data_start = min(9, len(raw)-1), min(18, len(raw)-1)
    titles = raw.iloc[hdr_row].astype(str).fillna("")
    data = raw.iloc[data_start:].reset_index(drop=True)

    def col_like(txt, must=None):
        mask = titles.str.contains(txt, case=False, regex=False)
        if must:
            for t in must:
                mask &= titles.str.contains(t, case=False, regex=False)
        idx = list(titles[mask].index)
        return data.columns[idx[0]] if idx and idx[0] < len(data.columns) else None

    cols = {
        "CuentaCorriente": col_like("Cuenta corriente") or 1,
        "Exportaciones": col_like("A. Bienes, Exportaciones de mer") or 4,
        "IngresosTurismo": col_like("B. Servicios", must=["Viajes", "I. Crédito"]) or 8,
        "Remesas": col_like("Remesas", must=["I. Crédito"]) or 18,
        "Importaciones": col_like("A. Bienes, Importaciones de mer") or 22,
        "BalanzaBienes": col_like("Balanza de bienes y servicios, Balanza de bienes") or 85,
        "BalanzaServicios": col_like("Balanza de bienes y servicios, Balanza de servicios") or 89,
        "BalanzaIngresoPrimario": col_like("Balanza de ingreso primario"),
        "BalanzaIngresoSecundario": col_like("Balanza de ingreso secundario"),
    }

    base = pd.DataFrame({"fecha": data.iloc[:, 0].apply(excel_to_datetime)})
    for k, v in cols.items():
        if v is not None and v in data.columns:
            base[k] = pd.to_numeric(data[v], errors="coerce")
        elif isinstance(v, int) and v < data.shape[1]:
            base[k] = pd.to_numeric(data.iloc[:, v], errors="coerce")
        else:
            base[k] = np.nan

    base = base.dropna(subset=["fecha"]).copy()
    base["anio"], base["trim"] = base["fecha"].dt.year, ((base["fecha"].dt.month - 1) // 3) + 1

    agg_cols = list(cols.keys())
    df_q = base.groupby(["anio", "trim"], as_index=False)[agg_cols].sum(min_count=1)
    df_q["periodo"] = df_q.apply(lambda r: periodo_trimestre(r["anio"], r["trim"]), axis=1)

    out = df_q[["periodo"] + agg_cols].copy()
    guardar_csv(out, RUTAS_SALIDA["balanza_pagos"], "Balanza de Pagos")
    exportar_validacion(validacion_basica(out, "balanza_pagos"), "balanza_pagos_validaciones.csv")
    return out


# ============================================================================
# SECCION 8: TASAS DE INTERES - BANXICO Y FED
# ============================================================================

@con_manejo_errores("Tasas de Interes")
def procesar_tasas_interes():
    """Procesa tasas de interes de Mexico y USA."""
    def to_quarter(df, date_col, value_col, out_name):
        s = df[[date_col, value_col]].copy()
        s[date_col] = pd.to_datetime(s[date_col], errors='coerce')
        s[value_col] = pd.to_numeric(s[value_col], errors='coerce')
        s = s.dropna().sort_values(date_col).set_index(date_col)[value_col]
        q = s.resample('Q').mean().to_frame(out_name)
        q.index = q.index.to_period('Q').to_timestamp(how='start')
        q = q.reset_index()
        q.columns = ['periodo', out_name]
        q['periodo'] = q['periodo'].dt.strftime('%Y-%m-%d')
        return q

    df_mx = pd.DataFrame()
    path_mx = RUTAS_ENTRADA["tasas_mx"]
    if path_mx.exists():
        try:
            # Leer con skiprows=17 para obtener header con códigos (Fecha, SF...)
            df = pd.read_excel(path_mx, sheet_name=0, skiprows=17, header=0)
            # SF61745 es la Tasa objetivo de Banxico
            tc = 'SF61745' if 'SF61745' in df.columns else None
            if not tc:
                # Buscar alternativa: última columna numérica o 'tasa objetivo'
                tc = next((c for c in df.columns if 'objetivo' in str(c).lower()), None)
            if tc and 'Fecha' in df.columns:
                tmp = df[['Fecha', tc]].copy()
                tmp = tmp[tmp[tc].astype(str).str.upper().ne('N/E')]
                tmp[tc] = pd.to_numeric(tmp[tc], errors='coerce')
                df_mx = to_quarter(tmp.dropna(), 'Fecha', tc, 'TasaMXN')
        except Exception as e:
            print(f"  ! Error cargando tasas MX: {e}")

    df_us = pd.DataFrame()
    path_us = RUTAS_ENTRADA["tasas_us"]
    if path_us.exists():
        try:
            df = pd.read_csv(path_us)
            if 'observation_date' in df.columns:
                df = df.rename(columns={'observation_date': 'DATE'})
            if 'DATE' in df.columns and 'FEDFUNDS' in df.columns:
                df = df[df['FEDFUNDS'].astype(str).str.upper().ne('N/A')].copy()
                df_us = to_quarter(df.dropna(), 'DATE', 'FEDFUNDS', 'TasaUSD')
        except:
            pass

    if df_mx.empty and df_us.empty:
        print("x No se pudieron cargar datos de tasas")
        return pd.DataFrame()

    if not df_mx.empty and not df_us.empty:
        out = pd.merge(df_mx, df_us, on='periodo', how='outer')
    else:
        out = df_mx if not df_mx.empty else df_us

    out = out.sort_values('periodo').reset_index(drop=True)

    # Redondeo como notebook original
    if 'TasaMXN' in out.columns:
        out['TasaMXN'] = out['TasaMXN'].round(4)
    if 'TasaUSD' in out.columns:
        out['TasaUSD'] = out['TasaUSD'].round(4)

    guardar_csv(out, RUTAS_SALIDA["tasas"], "Tasas de Interes")
    exportar_validacion(validacion_basica(out, "tasas"), "tasa_interes_validaciones.csv")
    return out


# ============================================================================
# SECCION 9: UNION Y CONVERSION DE DATASETS
# ============================================================================

def unir_y_convertir_datasets():
    """Une todos los datasets y genera versiones en USD corrientes y USD 2018."""
    print("\n" + "=" * 60 + "\nUNION Y CONVERSION DE DATASETS\n" + "=" * 60)

    # Cargar CPI
    path_cpi = RUTAS_ENTRADA["cpi"]
    if not path_cpi.exists():
        print("x No se encontro archivo CPI")
        return

    cpi = pd.read_csv(path_cpi)
    cpi['observation_date'] = pd.to_datetime(cpi['observation_date'])
    cpi['year'], cpi['quarter'] = cpi['observation_date'].dt.year, cpi['observation_date'].dt.quarter
    cpi_q = cpi.groupby(['year', 'quarter'])['CPIAUCSL'].mean().reset_index()
    cpi_q['periodo'] = cpi_q.apply(lambda r: periodo_trimestre(r['year'], r['quarter']), axis=1)
    cpi_2018 = cpi_q[cpi_q['year'] == 2018]['CPIAUCSL'].mean()
    cpi_q['CPI_base2018'] = (cpi_q['CPIAUCSL'] / cpi_2018) * 100
    print(f"+ CPI procesado, base 2018: {cpi_2018:.2f}")

    # Cargar datasets procesados
    def normalizar_periodo(df):
        df = df.copy()
        df['periodo'] = pd.to_datetime(df['periodo'], errors='coerce').dt.strftime('%Y-%m-%d')
        return df.dropna(subset=['periodo'])

    archivos = {
        'inflacion': RUTAS_SALIDA["inflacion"], 'balance_publico': RUTAS_SALIDA["balance_agg"],
        'deuda_externa': RUTAS_SALIDA["deuda_externa"], 'rfsp': RUTAS_SALIDA["rfsp"],
        'pib': RUTAS_SALIDA["pib"], 'tasa_interes': RUTAS_SALIDA["tasas"],
        'tipo_cambio': RUTAS_SALIDA["tipo_cambio"], 'balanza_pagos': RUTAS_SALIDA["balanza_pagos"]
    }
    datasets = {k: normalizar_periodo(pd.read_csv(v)) if v.exists() else pd.DataFrame()
                for k, v in archivos.items()}

    tcn_2018 = 19.65
    if not datasets['tipo_cambio'].empty:
        tc_2018 = datasets['tipo_cambio'][datasets['tipo_cambio']['periodo'].str.startswith('2018')]['tcn_mean']
        if not tc_2018.empty:
            tcn_2018 = tc_2018.mean()

    # Joins de CPI y TC
    cpi_join = cpi_q[['periodo', 'CPI_base2018']]
    tc_join = datasets['tipo_cambio'][['periodo', 'tcn_mean']] if not datasets['tipo_cambio'].empty else pd.DataFrame()

    for k in ['inflacion', 'balance_publico', 'deuda_externa', 'rfsp', 'balanza_pagos', 'pib']:
        if k in datasets and not datasets[k].empty:
            d = datasets[k].drop(columns=['tcn_mean', 'CPI_base2018'], errors='ignore')
            if not tc_join.empty:
                d = d.merge(tc_join, on='periodo', how='left')
            datasets[k] = d.merge(cpi_join, on='periodo', how='left')

    # Conversiones a USD
    if not datasets['pib'].empty:
        p = datasets['pib']
        if 'PIB_nom_trimestral' in p.columns and 'tcn_mean' in p.columns:
            p['PIB_USD_corriente'] = p['PIB_nom_trimestral'] / p['tcn_mean']
        if 'PIB_real_trimestral' in p.columns:
            p['PIB_USD_2018'] = p['PIB_real_trimestral'] / tcn_2018
        datasets['pib'] = p

    # Ambos balance_publico y rfsp: dividir /1000 y luego /tcn (replica notebook exactamente)
    # Nota: aunque balance_publico ya está en millones MXN, el notebook original
    # aplica /1000 adicional para obtener el mismo resultado
    for k, div in [('balance_publico', 1000), ('rfsp', 1000)]:
        if not datasets[k].empty:
            d = datasets[k]
            exclude = ['CICLO', 'TRIMESTRE', 'tcn_mean', 'CPI_base2018', 'periodo']
            for c in [x for x in d.select_dtypes(include=[np.number]).columns if x not in exclude]:
                if 'tcn_mean' in d.columns:
                    d[f"{c}_USD_corriente"] = (d[c] / div) / d['tcn_mean']
                if 'CPI_base2018' in d.columns and f"{c}_USD_corriente" in d.columns:
                    d[f"{c}_USD_2018"] = d[f"{c}_USD_corriente"] * (100 / d['CPI_base2018'])
            datasets[k] = d

    for k in ['deuda_externa', 'balanza_pagos']:
        if not datasets[k].empty:
            d = datasets[k]
            exclude = ['CICLO', 'TRIMESTRE', 'tcn_mean', 'CPI_base2018', 'periodo']
            for c in [x for x in d.select_dtypes(include=[np.number]).columns if x not in exclude]:
                if 'CPI_base2018' in d.columns:
                    d[f"{c}_2018"] = d[c] * (100 / d['CPI_base2018'])
            datasets[k] = d

    # Construir datasets finales
    if not datasets['inflacion'].empty:
        base = datasets['inflacion'][['periodo', 'year', 'quarter', 'inpc_q_eoq', 'inpc_q_avg',
                                       'inflacion_yoy_eoq', 'inflacion_yoy_avg']].copy()
    else:
        base = pd.DataFrame({'periodo': pd.date_range('2002-01-01', '2024-10-01', freq='QS').strftime('%Y-%m-%d')})

    ds_c, ds_2 = base.copy(), base.copy()

    def merge_cols(ds, df, cols):
        if not df.empty and cols:
            return ds.merge(df[['periodo'] + cols], on='periodo', how='left')
        return ds

    if not datasets['pib'].empty:
        ds_c = merge_cols(ds_c, datasets['pib'], [c for c in ['PIB_USD_corriente'] if c in datasets['pib'].columns])
        ds_2 = merge_cols(ds_2, datasets['pib'], [c for c in ['PIB_USD_2018'] if c in datasets['pib'].columns])

    for k in ['balance_publico', 'rfsp']:
        if not datasets[k].empty:
            ds_c = merge_cols(ds_c, datasets[k], [c for c in datasets[k].columns if 'USD_corriente' in c])
            ds_2 = merge_cols(ds_2, datasets[k], [c for c in datasets[k].columns if 'USD_2018' in c])

    for k in ['deuda_externa', 'balanza_pagos']:
        if not datasets[k].empty:
            exclude = ['periodo', 'CICLO', 'TRIMESTRE', 'tcn_mean', 'CPI_base2018']
            dc = [c for c in datasets[k].columns if c not in exclude and '_2018' not in c]
            d2 = [c for c in datasets[k].columns if '_2018' in c]
            ds_c = merge_cols(ds_c, datasets[k], dc)
            ds_2 = merge_cols(ds_2, datasets[k], d2)

    if not datasets['tasa_interes'].empty:
        ti = [c for c in ['TasaMXN', 'TasaUSD'] if c in datasets['tasa_interes'].columns]
        ds_c = merge_cols(ds_c, datasets['tasa_interes'], ti)
        ds_2 = merge_cols(ds_2, datasets['tasa_interes'], ti)

    if not datasets['tipo_cambio'].empty:
        ds_c = merge_cols(ds_c, datasets['tipo_cambio'], ['tcn_mean', 'tcn_eop'])
        ds_2 = merge_cols(ds_2, datasets['tipo_cambio'], ['tcn_mean', 'tcn_eop'])

    # Filtrar rango temporal
    ds_c = ds_c[(ds_c['periodo'] >= '2002-01-01') & (ds_c['periodo'] <= '2024-10-01')]
    ds_2 = ds_2[(ds_2['periodo'] >= '2002-01-01') & (ds_2['periodo'] <= '2024-10-01')]

    guardar_csv(ds_c, RUTAS_SALIDA["final_corrientes"], "Dataset Final USD Corrientes")
    guardar_csv(ds_2, RUTAS_SALIDA["final_2018"], "Dataset Final USD 2018")


# ============================================================================
# SECCION 10: LIMPIEZA FINAL
# ============================================================================

def limpiar_datasets_finales():
    """Aplica limpieza final a los datasets consolidados."""
    print("\n" + "=" * 60 + "\nLIMPIEZA FINAL\n" + "=" * 60)

    if not RUTAS_SALIDA["final_2018"].exists() or not RUTAS_SALIDA["final_corrientes"].exists():
        print("x No se encontraron los datasets finales")
        return

    DROP_BASE = ['CICLO', 'TRIMESTRE', 'Periodo', 'year', 'quarter',
                 'CICLO_rfsp', 'TRIMESTRE_rfsp', 'Periodo_rfsp',
                 'CICLO_de', 'TRIMESTRE_de', 'Periodo_de',
                 'year_pib', 'quarter_pib']

    # Columnas a eliminar para version reducida (como notebook original)
    DROP_2018 = [
        "inpc_q_eoq", "inpc_q_avg", "inflacion_yoy_eoq",
        "balance_derivado_mmxn_USD_2018", "gap_balance_mmxn_USD_2018",
        "rfsp_total_miles_mxn_USD_2018", "incurrimiento_neto_pasivos_miles_USD_2018",
        "adq_neta_activos_fin_miles_USD_2018", "tcn_eop",
        "CC_diff_2018", "rfsp_total_final_miles_USD_2018",
    ]

    DROP_CORR = [
        "inpc_q_eoq", "inpc_q_avg", "inflacion_yoy_eoq",
        "balance_derivado_mmxn_USD_corriente", "gap_balance_mmxn_USD_corriente",
        "rfsp_total_miles_mxn_USD_corriente", "incurrimiento_neto_pasivos_miles_USD_corriente",
        "adq_neta_activos_fin_miles_USD_corriente", "tcn_eop",
        "CC_diff", "rfsp_total_final_miles_USD_corriente",
    ]

    def limpiar(df, extra_drop=None):
        d = df.copy()
        # Eliminar columnas con >50% nulos
        nulos = d.isnull().sum() / len(d) * 100
        d = d.drop(columns=nulos[nulos > 50].index.tolist(), errors='ignore')
        d = d.drop(columns=[c for c in DROP_BASE if c in d.columns], errors='ignore')
        # Imputar nulos con media
        for c in d.select_dtypes(include=[np.number]).columns:
            if d[c].isnull().sum():
                d[c] = d[c].fillna(d[c].mean())
        # Reordenar columnas
        lead = [c for c in ['periodo'] if c in d.columns]
        return d[lead + [c for c in d.columns if c not in lead]]

    def reducir(df, drop_list):
        return df.drop(columns=[c for c in drop_list if c in df.columns], errors='ignore')

    df_2018 = limpiar(pd.read_csv(RUTAS_SALIDA["final_2018"]))
    df_corr = limpiar(pd.read_csv(RUTAS_SALIDA["final_corrientes"]))

    guardar_csv(df_2018, RUTAS_SALIDA["final_2018_limpio"], "Dataset 2018 Limpio")
    guardar_csv(df_corr, RUTAS_SALIDA["final_corrientes_limpio"], "Dataset Corrientes Limpio")

    # Versiones reducidas (sin columnas redundantes)
    df_2018_red = reducir(df_2018, DROP_2018)
    df_corr_red = reducir(df_corr, DROP_CORR)

    guardar_csv(df_2018_red, RUTAS_SALIDA["final_2018_reducido"], "Dataset 2018 Reducido")
    guardar_csv(df_corr_red, RUTAS_SALIDA["final_corrientes_reducido"], "Dataset Corrientes Reducido")


# ============================================================================
# SECCION 11: INDICE FSI
# ============================================================================

def crear_fsi():
    """Crea el FSI - Indice de Sostenibilidad Financiera."""
    print("\n" + "=" * 60 + "\nCREACION FSI\n" + "=" * 60)

    WEIGHTS = {"solvencia": 0.30, "externa": 0.20, "mercado": 0.20, "liquidez": 0.15, "fiscal": 0.15}

    def pick(df, cs):
        return next((c for c in cs if c in df.columns), None)

    def sdiv(n, d):
        o = pd.to_numeric(n, errors="coerce") / pd.to_numeric(d, errors="coerce")
        o.replace([np.inf, -np.inf], np.nan, inplace=True)
        return o

    def pct_score(s, higher_is_better):
        r = s.rank(pct=True)
        return ((r if higher_is_better else (1 - r)) * 100).astype(float) if not s.dropna().empty else pd.Series(np.nan, index=s.index)

    def hhi(df, cols):
        """Calcula indice Herfindahl-Hirschman para concentracion."""
        ok = [c for c in cols if c in df.columns]
        if not ok:
            return pd.Series(np.nan, index=df.index)
        data = df[ok].apply(pd.to_numeric, errors="coerce")
        shares = data.div(data.sum(axis=1), axis=0)
        return (shares ** 2).sum(axis=1)

    def compute_fsi(df_in, tag, out_dir):
        df = df_in.copy()
        if "periodo" in df.columns:
            df["periodo"] = pd.to_datetime(df["periodo"], errors="coerce")
            df = df.sort_values("periodo").reset_index(drop=True)

        # Seleccionar columnas disponibles
        pib = pick(df, ["PIB_USD_2018", "PIB_USD_corriente"])
        dt = pick(df, ["deuda_bruta_publica_musd_2018", "deuda_bruta_publica_musd"])
        dlp = pick(df, ["deuda_publica_lp_musd_2018", "deuda_publica_lp_musd"])
        dcp = pick(df, ["deuda_publica_cp_musd_2018", "deuda_publica_cp_musd"])
        en = pick(df, ["endeudamiento_neto_total_musd_2018", "endeudamiento_neto_total_musd"])
        cc = pick(df, ["CuentaCorriente_2018", "CuentaCorriente"])
        bi = pick(df, ["BalanzaBienes_2018", "BalanzaBienes"])
        ip = pick(df, ["BalanzaIngresoPrimario_2018", "BalanzaIngresoPrimario"])
        iy = pick(df, ["inflacion_yoy_avg"])
        tc = pick(df, ["tcn_mean"])
        tmx = pick(df, ["TasaMXN"])
        tus = pick(df, ["TasaUSD"])
        bp = pick(df, ["balance_mmxn_USD_2018", "balance_mmxn_USD_corriente"])
        rf = pick(df, ["rfsp_sin_ingresos_rec_miles_USD_2018", "rfsp_sin_ingresos_rec_miles_USD_corriente"])
        mu = pick(df, ["moneda_usd_musd_2018", "moneda_usd_musd"])

        # Columnas de acreedores y fuentes para HHI
        ac = [c for c in df.columns if c.startswith("acreedor_") and c.endswith(("_musd_2018", "_musd"))]
        fc = [c for c in df.columns if c.startswith("fuente_") and c.endswith(("_musd_2018", "_musd"))]

        # Construir componentes
        comp = pd.DataFrame({
            "periodo": df.get("periodo", pd.NA),
            "debt_to_gdp": sdiv(df[dt], df[pib]) if dt and pib else np.nan,
            "netdebt_to_gdp": sdiv(df[en], df[pib]) if en and pib else np.nan,
            "lt_share": sdiv(df[dlp], df[dlp] + df[dcp]) if dlp and dcp else np.nan,
            "ca_to_gdp": sdiv(df[cc], df[pib]) if cc and pib else np.nan,
            "goods_to_gdp": sdiv(df[bi], df[pib]) if bi and pib else np.nan,
            "priminc_to_gdp": sdiv(df[ip], df[pib]) if ip and pib else np.nan,
            "infl_yoy": df[iy] if iy else np.nan,
            "fx_dep_yoy": pd.to_numeric(df[tc], errors="coerce").pct_change(4) if tc else np.nan,
            "rate_mxn": df[tmx] if tmx else np.nan,
            "rate_usd": df[tus] if tus else np.nan,
            "st_share": sdiv(df[dcp], df[dlp] + df[dcp]) if dlp and dcp else np.nan,
            "usd_share": sdiv(df[mu], df[dt]) if mu and dt else np.nan,
            "hhi_acreedores": hhi(df, ac),
            "hhi_fuentes": hhi(df, fc),
            "fiscal_to_gdp": sdiv(df[bp], df[pib]) if bp and pib else ((-sdiv(df[rf], df[pib])) if rf and pib else np.nan),
        })

        # Direcciones: True = mayor es mejor, False = menor es mejor
        dirs = {
            "debt_to_gdp": False, "netdebt_to_gdp": False, "lt_share": True,
            "ca_to_gdp": True, "goods_to_gdp": True, "priminc_to_gdp": True,
            "infl_yoy": False, "fx_dep_yoy": False, "rate_mxn": False, "rate_usd": False,
            "st_share": False, "usd_share": False, "hhi_acreedores": False, "hhi_fuentes": False,
            "fiscal_to_gdp": True
        }

        scores = pd.DataFrame({k + "_score": pct_score(comp[k], dirs[k]) for k in comp.columns if k in dirs})

        # Pilares con todas las variables
        pillars = {
            "solvencia": ["debt_to_gdp_score", "netdebt_to_gdp_score", "lt_share_score"],
            "externa": ["ca_to_gdp_score", "goods_to_gdp_score", "priminc_to_gdp_score"],
            "mercado": ["infl_yoy_score", "fx_dep_yoy_score", "rate_mxn_score", "rate_usd_score"],
            "liquidez": ["st_share_score", "usd_share_score", "hhi_acreedores_score", "hhi_fuentes_score"],
            "fiscal": ["fiscal_to_gdp_score"]
        }

        pillar_df = pd.DataFrame()
        for p, cs in pillars.items():
            valid_cols = [c for c in cs if c in scores.columns and not scores[c].dropna().empty]
            if valid_cols:
                pillar_df[p] = scores[valid_cols].mean(axis=1)
            else:
                pillar_df[p] = np.nan

        def calc_fsi(row):
            available = {p: row[p] for p in WEIGHTS if pd.notna(row[p])}
            if not available:
                return np.nan
            total_weight = sum(WEIGHTS[p] for p in available)
            return sum(row[p] * WEIGHTS[p] / total_weight for p in available)

        fsi = pillar_df.apply(calc_fsi, axis=1)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Guardar componentes completos
        try:
            pd.concat([comp, scores, pillar_df.rename(columns=lambda c: f"p_{c}")], axis=1).to_csv(
                out_dir / f"fsi_components_{tag}.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
        except PermissionError:
            print(f"  ! No se pudo escribir fsi_components_{tag}.csv (archivo bloqueado)")

        # Guardar serie temporal
        result = pd.DataFrame({
            "periodo": comp["periodo"], "FSI": fsi.round(2),
            **{f"p_{p}": pillar_df[p].round(2) for p in pillars}
        })
        try:
            guardar_csv(result, out_dir / f"fsi_timeseries_{tag}.csv", f"FSI {tag}")
        except PermissionError:
            print(f"  ! No se pudo escribir fsi_timeseries_{tag}.csv (archivo bloqueado)")

        # Guardar pesos
        try:
            with open(out_dir / f"fsi_weights_{tag}.json", "w", encoding="utf-8") as f:
                json.dump({"tag": tag, "weights": WEIGHTS}, f, ensure_ascii=False, indent=2)
        except PermissionError:
            print(f"  ! No se pudo escribir fsi_weights_{tag}.json (archivo bloqueado)")

    INDICADORES_DIR.mkdir(parents=True, exist_ok=True)

    if RUTAS_SALIDA["final_2018_reducido"].exists():
        compute_fsi(pd.read_csv(RUTAS_SALIDA["final_2018_reducido"]), "USD_2018", INDICADORES_DIR)
    if RUTAS_SALIDA["final_corrientes_reducido"].exists():
        compute_fsi(pd.read_csv(RUTAS_SALIDA["final_corrientes_reducido"]), "USD_Corrientes", INDICADORES_DIR)


# ============================================================================
# EJECUCION PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60 + "\nPIPELINE ETL - DATOS FISCALES DE MEXICO\n" + "=" * 60 + "\n")

    inicializar_directorios()

    # Procesar fuentes de datos
    procesar_deuda_externa()
    procesar_balance_publico()
    procesar_rfsp()
    procesar_pib()
    procesar_inflacion()
    procesar_tipo_cambio()
    procesar_balanza_pagos()
    procesar_tasas_interes()

    # Unir y limpiar
    unir_y_convertir_datasets()
    limpiar_datasets_finales()
    crear_fsi()

    print("\n" + "=" * 60 + "\nPIPELINE COMPLETADO\n" + "=" * 60)
