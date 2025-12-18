# -*- coding: utf-8 -*-
"""
Dashboard Deuda Externa - México
Dashboard interactivo para análisis de sostenibilidad fiscal
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys

# Agregar directorio actual al path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

# Ruta de imágenes
IMAGES_DIR = BASE_DIR / "Imagenes"

from procesamiento import ProcesadorDatos, ejecutar_procesamiento

# =============================================================================
# CONFIGURACIÓN DE PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Dashboard Deuda Externa - México",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# PALETA DE COLORES
# =============================================================================
COLORES = {
    'primario': '#2E7D32',        # Verde indicador
    'secundario': '#E53935',      # Rojo indicador
    'fondo': '#F5F9FC',           # Azul cielo casi blanco
    'fondo_card': '#FFFFFF',      # Blanco
    'texto': '#000000',           # Negro
    'positivo': '#2E7D32',        # Verde (indicadores positivos)
    'negativo': '#E53935',        # Rojo (indicadores negativos)
    'neutro': '#FF9800',          # Naranja
    'gradiente_inicio': '#2E7D32',
    'gradiente_fin': '#4CAF50'
}

# =============================================================================
# ESTILOS CSS
# =============================================================================
st.markdown("""
<style>
    :root{
        --bg: #F5F9FC;
        --bg2: #EDF4F8;
        --card: #FFFFFF;
        --text: #000000;
        --text-secondary: #333333;
        --border: #D0E0EB;
        --verde: #2E7D32;
        --rojo: #E53935;
        --shadow: 0 4px 12px rgba(0,0,0,0.08);
        --shadow-sm: 0 2px 6px rgba(0,0,0,0.05);
    }

    /* Tipografía y fondo */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', system-ui, -apple-system, Roboto, Helvetica, Arial, sans-serif;
        color: var(--text) !important;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    /* Separadores */
    hr { border-top: 1px solid var(--border) !important; margin: 1.5rem 0 !important; }

    /* ============ SIDEBAR - FONDO BEIGE ============ */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div,
    [data-testid="stSidebar"] > div > div,
    [data-testid="stSidebar"] section,
    [data-testid="stSidebar"] section > div {
        background: #F5EDE3 !important;
        background-color: #F5EDE3 !important;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: var(--text) !important;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: var(--text) !important;
        font-weight: 600;
    }

    /* Selectbox en sidebar */
    [data-testid="stSidebar"] .stSelectbox label {
        color: var(--text) !important;
        font-weight: 700 !important;
        font-size: 0.95rem;
    }

    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        background: var(--card) !important;
    }

    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
        background: var(--card) !important;
        border: 1.5px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
    }

    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] span,
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] div {
        color: var(--text) !important;
    }

    /* Info box en sidebar */
    [data-testid="stSidebar"] .stAlert {
        background: rgba(30,136,229,0.08) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px;
    }

    [data-testid="stSidebar"] .stAlert p {
        color: var(--text) !important;
        font-size: 0.85rem;
    }

    /* ============ WIDGETS GENERALES ============ */
    .stSelectbox div[data-baseweb="select"] > div,
    .stTextInput input,
    .stNumberInput input {
        background: var(--card) !important;
        border: 1.5px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
    }

    .stSelectbox div[data-baseweb="select"] > div:hover {
        border-color: var(--verde) !important;
    }

    .stSelectbox div[data-baseweb="select"] > div:focus-within {
        border-color: var(--verde) !important;
        box-shadow: 0 0 0 3px rgba(46,125,50,0.15) !important;
    }

    /* ============ TÍTULO PRINCIPAL ============ */
    .main-title {
        color: var(--text) !important;
        font-size: 2.2rem;
        font-weight: 800;
        text-align: center;
        margin: 0.5rem 0 1.5rem;
    }

    /* ============ KPI BOXES ============ */
    .kpi-box {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: var(--shadow-sm);
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        margin-top: 8px;
    }
    .kpi-label {
        font-size: 0.9rem;
        color: var(--text) !important;
        margin-top: 6px;
        font-weight: 600;
    }

    /* ============ INDICADORES MACRO ============ */
    .macro-indicator {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 12px 14px;
        margin: 8px 0;
        box-shadow: var(--shadow-sm);
    }
    .macro-indicator strong {
        color: var(--text) !important;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .macro-indicator span {
        color: var(--text) !important;
    }

    /* ============ CAJA ECUACIÓN ============ */
    .equation-box {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        font-family: 'Consolas', 'Monaco', monospace;
        color: var(--text) !important;
        box-shadow: var(--shadow-sm);
    }
    .equation-box strong, .equation-box span, .equation-box small {
        color: var(--text) !important;
    }

    /* ============ STATUS PILL ============ */
    .status-pill {
        padding: 14px;
        border-radius: 12px;
        text-align: center;
        box-shadow: var(--shadow-sm);
        font-weight: 700;
    }

    /* ============ MENÚ DESPLEGABLE (DROPDOWN) - FONDO BEIGE ============ */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    [data-baseweb="popover"] {
        background-color: #F5EDE3 !important;
    }
    div[data-baseweb="popover"] ul,
    [data-baseweb="popover"] ul {
        background-color: #F5EDE3 !important;
    }
    div[data-baseweb="popover"] li,
    [data-baseweb="popover"] li,
    [data-baseweb="popover"] [role="option"] {
        background-color: #F5EDE3 !important;
        color: #000000 !important;
    }
    div[data-baseweb="popover"] li:hover,
    [data-baseweb="popover"] li:hover,
    [data-baseweb="popover"] [role="option"]:hover,
    [data-baseweb="popover"] [aria-selected="true"] {
        background-color: #EDE5D8 !important;
    }
    ul[data-testid="stSelectboxVirtualDropdown"],
    [data-testid="stSelectboxVirtualDropdown"] {
        background-color: #F5EDE3 !important;
    }
    ul[data-testid="stSelectboxVirtualDropdown"] li,
    [data-testid="stSelectboxVirtualDropdown"] li {
        background-color: #F5EDE3 !important;
        color: #000000 !important;
    }
    ul[data-testid="stSelectboxVirtualDropdown"] li:hover,
    [data-testid="stSelectboxVirtualDropdown"] li:hover {
        background-color: #EDE5D8 !important;
    }
    /* Opciones del menú */
    [data-baseweb="menu"],
    [data-baseweb="menu"] > div {
        background-color: #F5EDE3 !important;
    }
    [data-baseweb="menu"] li,
    [data-baseweb="menu"] [role="option"] {
        background-color: #F5EDE3 !important;
        color: #000000 !important;
    }
    [data-baseweb="menu"] li:hover,
    [data-baseweb="menu"] [role="option"]:hover {
        background-color: #EDE5D8 !important;
    }
    /* Lista desplegable del select */
    [data-baseweb="select"] [data-baseweb="popover"],
    [data-baseweb="select"] ul {
        background-color: #F5EDE3 !important;
    }

    /* ============ ST.TABLE - ESTILOS ============ */
    .stTable,
    [data-testid="stTable"] {
        background-color: #FFFFFF !important;
        border-radius: 10px !important;
        overflow: hidden;
    }
    .stTable table,
    [data-testid="stTable"] table {
        background-color: #FFFFFF !important;
        border-collapse: collapse !important;
        width: 100% !important;
    }
    .stTable th,
    [data-testid="stTable"] th {
        background-color: #F0F0F0 !important;
        color: #000000 !important;
        font-weight: 700 !important;
        padding: 10px 12px !important;
        border-bottom: 2px solid #D0E0EB !important;
    }
    .stTable td,
    [data-testid="stTable"] td {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        padding: 8px 12px !important;
        border-bottom: 1px solid #E0E0E0 !important;
    }
    .stTable tr:nth-child(even) td,
    [data-testid="stTable"] tr:nth-child(even) td {
        background-color: #FAFAFA !important;
    }

    /* ============ EXPANDERS - FONDO BLANCO (SIN AZUL OSCURO) ============ */
    .stExpander,
    [data-testid="stExpander"],
    details.streamlit-expanderHeader,
    div[data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        overflow: hidden;
    }
    .stExpander > div,
    [data-testid="stExpander"] > div,
    .stExpander > details,
    [data-testid="stExpander"] > details {
        background-color: #FFFFFF !important;
    }
    .stExpander summary,
    [data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary {
        background-color: #FFFFFF !important;
        padding: 0.8rem 1rem;
    }
    .stExpander summary span,
    .stExpander summary p,
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary p {
        color: var(--text) !important;
        font-weight: 700 !important;
    }
    .stExpander div[data-testid="stExpanderDetails"],
    [data-testid="stExpander"] div[data-testid="stExpanderDetails"],
    div[data-testid="stExpanderDetails"] {
        background-color: #FFFFFF !important;
        padding: 1rem;
        color: var(--text) !important;
    }
    .stExpander div[data-testid="stExpanderDetails"] *,
    [data-testid="stExpander"] div[data-testid="stExpanderDetails"] *,
    div[data-testid="stExpanderDetails"] * {
        background-color: transparent !important;
        color: var(--text) !important;
    }
    .stExpander div[data-testid="stExpanderDetails"] pre,
    .stExpander div[data-testid="stExpanderDetails"] code,
    div[data-testid="stExpanderDetails"] pre,
    div[data-testid="stExpanderDetails"] code {
        background-color: #F8F8F8 !important;
        color: var(--text) !important;
    }
    /* Forzar fondo blanco en todo el contenedor del expander */
    .stExpander [data-testid="stMarkdownContainer"],
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] {
        background-color: transparent !important;
    }
    /* Eliminar cualquier fondo oscuro en el header del expander */
    .streamlit-expanderHeader,
    [data-testid="stExpander"] > div:first-child {
        background-color: #FFFFFF !important;
    }

    /* ============ DATAFRAMES / TABLAS - FONDO BLANCO (SIN AZUL OSCURO) ============ */
    div[data-testid="stDataFrame"],
    [data-testid="stDataFrame"],
    .stDataFrame {
        background-color: #FFFFFF !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        padding: 8px !important;
        box-shadow: var(--shadow-sm);
    }
    div[data-testid="stDataFrame"] > div,
    [data-testid="stDataFrame"] > div,
    .stDataFrame > div {
        background-color: #FFFFFF !important;
    }
    div[data-testid="stDataFrame"] iframe,
    [data-testid="stDataFrame"] iframe {
        background-color: #FFFFFF !important;
    }
    /* Contenedor del dataframe */
    [data-testid="stDataFrame"] [data-testid="stDataFrameResizable"],
    div[data-testid="stDataFrameResizable"] {
        background-color: #FFFFFF !important;
    }
    /* Glide Data Grid (nuevo componente de tabla en Streamlit) */
    [data-testid="stDataFrame"] .dvn-scroller,
    .dvn-scroller {
        background-color: #FFFFFF !important;
    }
    [data-testid="stDataFrame"] canvas + div,
    .gdg-style {
        background-color: #FFFFFF !important;
    }

    .stDataFrame table {
        border-collapse: collapse !important;
        background-color: #FFFFFF !important;
    }

    .stDataFrame th {
        background-color: #F0F0F0 !important;
        color: var(--text) !important;
        font-weight: 700 !important;
        padding: 10px 12px !important;
        text-align: left !important;
        font-size: 0.85rem !important;
    }

    .stDataFrame td {
        background-color: #FFFFFF !important;
        color: var(--text) !important;
        padding: 8px 12px !important;
        border-bottom: 1px solid var(--border) !important;
        font-size: 0.85rem !important;
    }

    .stDataFrame tr:nth-child(even) td {
        background-color: #FAFAFA !important;
    }

    .stDataFrame tr:hover td {
        background-color: #F5F5F5 !important;
    }

    /* Forzar tema claro en la tabla glide-data-grid */
    [data-testid="stDataFrame"] [class*="glideDataEditor"],
    [class*="glideDataEditor"] {
        --gdg-bg-cell: #FFFFFF !important;
        --gdg-bg-header: #F0F0F0 !important;
        --gdg-text-dark: #000000 !important;
        --gdg-text-medium: #333333 !important;
    }

    /* ============ MÉTRICAS STREAMLIT ============ */
    [data-testid="stMetricValue"] {
        color: var(--text) !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricDelta"] {
        font-weight: 700 !important;
    }

    /* ============ HEADINGS ============ */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text) !important;
    }

    .stMarkdown h3 {
        color: var(--text) !important;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 0.75rem;
    }

    /* ============ ALERTS / INFO ============ */
    .stAlert {
        background: rgba(139,90,60,0.06) !important;
        border: 1px solid var(--accent-light) !important;
        border-radius: 10px !important;
    }
    .stAlert p {
        color: var(--text) !important;
    }

    /* ============ PLOTLY CHARTS ============ */
    .js-plotly-plot .plotly .modebar {
        background: transparent !important;
    }

</style>
""", unsafe_allow_html=True)


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

@st.cache_resource(ttl=3600)
def cargar_datos(proceso='base_2018', tipo_modelo='series_tiempo'):
    """Carga y cachea los datos del procesamiento"""
    procesador = ejecutar_procesamiento(proceso=proceso, tipo_modelo=tipo_modelo)
    return procesador


@st.cache_data(ttl=3600)
def cargar_datos_secundarios():
    """Carga los datos secundarios para comparación"""
    base_path = Path(__file__).parent
    path_secundarios = base_path / "Datos_Resultado" / "DatasetSecundarioFinal"

    # Verificar si el directorio existe
    if not path_secundarios.exists():
        return None, None

    # Verificar si los archivos existen
    file_pib = path_secundarios / "PIB_Corriente_BancoMundial_Mexico.csv"
    file_deuda = path_secundarios / "DeudaExterna_Banxico_Trimestral.csv"

    if not file_pib.exists() or not file_deuda.exists():
        return None, None

    # Cargar PIB Banco Mundial
    df_pib_bm = pd.read_csv(file_pib)

    # Cargar Deuda Externa Banxico (trimestral)
    df_deuda_banxico = pd.read_csv(file_deuda, parse_dates=['Fecha'])
    df_deuda_banxico.set_index('Fecha', inplace=True)

    return df_pib_bm, df_deuda_banxico


def calcular_metricas_similitud(serie_modelo, serie_externa, nombre=''):
    """Calcula métricas de similitud entre dos series"""
    # Alinear índices
    idx_comun = serie_modelo.index.intersection(serie_externa.index)
    if len(idx_comun) < 2:
        return None

    s1 = serie_modelo.loc[idx_comun].dropna()
    s2 = serie_externa.loc[idx_comun].dropna()

    # Recalcular índice común después de dropna
    idx_final = s1.index.intersection(s2.index)
    if len(idx_final) < 2:
        return None

    s1 = s1.loc[idx_final]
    s2 = s2.loc[idx_final]

    # Correlación
    correlacion = s1.corr(s2)

    # RMSE
    rmse = np.sqrt(np.mean((s1 - s2) ** 2))

    # MAPE
    mape = np.mean(np.abs((s1 - s2) / s1)) * 100

    # MAE
    mae = np.mean(np.abs(s1 - s2))

    # Diferencia promedio
    diff_prom = np.mean(s2 - s1)

    # Formatear rango de fechas
    fecha_min = idx_final.min()
    fecha_max = idx_final.max()
    rango = f"{fecha_min.year}-Q{(fecha_min.month-1)//3+1} a {fecha_max.year}-Q{(fecha_max.month-1)//3+1}"

    return {
        'nombre': nombre,
        'n_observaciones': len(idx_final),
        'correlacion': correlacion,
        'rmse': rmse,
        'mape': mape,
        'mae': mae,
        'diff_promedio': diff_prom,
        'rango_fechas': rango
    }


def filtrar_datos(df, año, trimestre):
    """Aplica filtros de año y trimestre a un DataFrame"""
    df_filtrado = df.copy()
    if año != 'Todos':
        df_filtrado = df_filtrado[df_filtrado.index.year == año]
    if trimestre != 'Todos':
        q_map = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}
        df_filtrado = df_filtrado[df_filtrado.index.quarter == q_map[trimestre]]
    return df_filtrado


def crear_kpi_card(valor, etiqueta, delta=None, delta_color='normal'):
    """Crea una tarjeta KPI estilizada"""
    delta_html = ""
    if delta is not None:
        color = '#E74C3C' if delta > 0 else '#27AE60'
        simbolo = '▲' if delta > 0 else '▼'
        delta_html = f'<span style="color: {color}; font-size: 0.9rem;">{simbolo} {abs(delta):.2f}%</span>'

    return f"""
    <div class="kpi-box">
        <div class="kpi-value">{valor}</div>
        <div class="kpi-label">{etiqueta}</div>
        {delta_html}
    </div>
    """


def crear_indicador_macro(nombre, valor, tendencia=None):
    """Crea un indicador macroeconómico"""
    trend_html = ""
    if tendencia is not None:
        color = '#27AE60' if tendencia > 0 else '#E74C3C'
        simbolo = '▲' if tendencia > 0 else '▼'
        trend_html = f'<span style="color: {color}; font-size: 0.8rem;"> {simbolo}</span>'

    return f"""
    <div class="macro-indicator">
        <strong>{nombre}</strong>{trend_html}<br>
        <span style="font-size: 1.25rem; color: {COLORES['texto']}; font-weight: 400;">{valor}</span>
    </div>
    """


def formato_numero(valor, decimales=2, sufijo=''):
    """Formatea números para visualización"""
    if pd.isna(valor):
        return 'N/A'
    if abs(valor) >= 1e9:
        return f"{valor/1e9:.{decimales}f}B{sufijo}"
    elif abs(valor) >= 1e6:
        return f"{valor/1e6:.{decimales}f}M{sufijo}"
    elif abs(valor) >= 1e3:
        return f"{valor/1e3:.{decimales}f}K{sufijo}"
    else:
        return f"{valor:.{decimales}f}{sufijo}"


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    # =============================================================================
    # SIDEBAR (primero para obtener filtros)
    # =============================================================================
    with st.sidebar:
        # Logo
        logo_path = Path(__file__).parent / "Imagenes" / "Foca_logo.png"
        if logo_path.exists():
            st.image(str(logo_path), width=150)
        else:
            # Intenta ruta alternativa
            logo_alt = Path(__file__).parent / "Foca_logo.png"
            if logo_alt.exists():
                st.image(str(logo_alt), width=150)
            else:
                st.markdown("### 🦭")

        st.markdown("---")
        st.markdown("## Dashboard Deuda Externa")
        st.markdown("### México")
        st.markdown("---")

        # Filtros
        st.markdown("### 📅 Filtros")

        # Selector de proceso (tipo de datos)
        proceso_opciones = {
            'Base 2018': 'base_2018',
            'Cuenta Corriente': 'cuenta_corriente',
            'Comparativa Banxico': 'comparativa_corriente'
        }
        proceso_seleccionado_label = st.selectbox(
            "Proceso",
            options=list(proceso_opciones.keys()),
            index=0,
            help="Base 2018: Datos a precios constantes 2018\nCuenta Corriente: Datos a precios corrientes\nComparativa Banxico: Compara modelo con datos reales"
        )
        proceso_seleccionado = proceso_opciones[proceso_seleccionado_label]

        # Determinar si estamos en modo comparativa
        modo_comparativa = (proceso_seleccionado == 'comparativa_corriente')

        # Selector de tipo de modelo (solo si NO es modo comparativa)
        modelo_opciones = {
            'Series de Tiempo': 'series_tiempo',
            'Machine Learning': 'machine_learning',
            'Híbrido (ST + ML)': 'hibrido'
        }

        if not modo_comparativa:
            tipo_modelo_label = st.selectbox(
                "Modelo",
                options=list(modelo_opciones.keys()),
                index=0,
                help="Series de Tiempo: ARIMA, SARIMA, ETS\nMachine Learning: Ridge, Random Forest, Gradient Boosting\nHíbrido: Combinación 50% ST + 50% ML"
            )
            tipo_modelo = modelo_opciones[tipo_modelo_label]
        else:
            tipo_modelo = 'series_tiempo'
            tipo_modelo_label = 'Series de Tiempo'
            st.info("📊 Modo Comparativa: se comparan datos del modelo con datos reales de Banxico")

    # Cargar datos según el proceso y modelo seleccionado
    proceso_para_cargar = 'cuenta_corriente' if proceso_seleccionado == 'comparativa_corriente' else proceso_seleccionado
    procesador = cargar_datos(proceso=proceso_para_cargar, tipo_modelo=tipo_modelo)
    datos = procesador.obtener_datos_dashboard()
    ultimo = procesador.obtener_ultimo_periodo()
    ecuacion = procesador.obtener_ecuacion_modelo()

    # Cargar datos secundarios si es modo comparativa corriente
    df_pib_bm, df_deuda_banxico = None, None
    if proceso_seleccionado == 'comparativa_corriente':
        try:
            df_pib_bm, df_deuda_banxico = cargar_datos_secundarios()
        except Exception as e:
            st.error(f"Error cargando datos secundarios: {e}")

    # Continuar con sidebar
    with st.sidebar:
        # Selector de año
        años_disponibles = sorted(datos['df_analisis'].index.year.unique())
        año_seleccionado = st.selectbox(
            "Año",
            options=['Todos'] + list(años_disponibles),
            index=0
        )

        # Selector de trimestre
        trimestres = ['Todos', 'Q1', 'Q2', 'Q3', 'Q4']
        trimestre_seleccionado = st.selectbox(
            "Trimestre",
            options=trimestres,
            index=0
        )

        # Tabla de umbrales de referencia (solo en modo predicción, no comparativa)
        if not modo_comparativa:
            st.markdown("---")
            st.markdown("**Umbrales de referencia Deuda/PIB**")
            st.markdown("<small>*(Estándares internacionales)*</small>", unsafe_allow_html=True)
            st.markdown("""
            <table style="width: 100%; font-size: 0.8rem; border-collapse: collapse; background: #FFFFFF;">
                <tr style="background: #F0F0F0;">
                    <th style="padding: 6px 8px; text-align: left; border-bottom: 2px solid #D0E0EB;">Rango</th>
                    <th style="padding: 6px 8px; text-align: left; border-bottom: 2px solid #D0E0EB;">Clasificación</th>
                </tr>
                <tr style="background: #FFFFFF;">
                    <td style="padding: 5px 8px; border-bottom: 1px solid #E0E0E0;">< 30%</td>
                    <td style="padding: 5px 8px; border-bottom: 1px solid #E0E0E0;">🟢 Bajo</td>
                </tr>
                <tr style="background: #FFFFFF;">
                    <td style="padding: 5px 8px; border-bottom: 1px solid #E0E0E0;">30-50%</td>
                    <td style="padding: 5px 8px; border-bottom: 1px solid #E0E0E0;">🟡 Moderado</td>
                </tr>
                <tr style="background: #FFFFFF;">
                    <td style="padding: 5px 8px; border-bottom: 1px solid #E0E0E0;">50-70%</td>
                    <td style="padding: 5px 8px; border-bottom: 1px solid #E0E0E0;">🟠 Alto</td>
                </tr>
                <tr style="background: #FFFFFF;">
                    <td style="padding: 5px 8px;"><b>> 70%</b></td>
                    <td style="padding: 5px 8px;">🔴 Crítico</td>
                </tr>
            </table>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Última actualización de los datos:**")
        st.markdown(f"{ultimo['fecha'].strftime('%Y-%m-%d')}")
        st.markdown(f"**Proceso:** {proceso_seleccionado_label}")
        if not modo_comparativa:
            st.markdown(f"**Modelo:** {tipo_modelo_label}")

        # Escudo de México en sidebar
        st.markdown("---")
        st.image(str(IMAGES_DIR / "escudo_mexico.png"), use_container_width=True)

    # =============================================================================
    # CONTENIDO PRINCIPAL
    # =============================================================================

    # Logos en encabezado (IPN izquierda, ESCOM derecha)
    col_logo_izq, col_titulo, col_logo_der = st.columns([1, 6, 1])
    with col_logo_izq:
        st.image(str(IMAGES_DIR / "IPN-Logo.png"), width=80)
    with col_titulo:
        # Título - varía según el modo
        if modo_comparativa:
            st.markdown('<h1 class="main-title">Comparativa: Modelo vs Datos Reales Banxico</h1>', unsafe_allow_html=True)
        else:
            st.markdown('<h1 class="main-title">Dashboard Deuda Externa</h1>', unsafe_allow_html=True)
    with col_logo_der:
        st.image(str(IMAGES_DIR / "Escom-Logo.png"), width=80)

    # =============================================================================
    # CONTENIDO CONDICIONAL: MODO PREDICCIÓN vs MODO COMPARATIVA
    # =============================================================================

    if not modo_comparativa:
        # =============================================================================
        # LAYOUT MODO PREDICCIÓN (Series de Tiempo, ML o Híbrido)
        # =============================================================================

        # =============================================================================
        # FILA SUPERIOR: Ecuación del Modelo + KPIs
        # =============================================================================
        col_eq, col_kpi1, col_kpi2 = st.columns([2, 1, 1])

        with col_eq:
            st.markdown(f"### 📐 Modelo: {ecuacion['nombre']}")

            # Obtener valores reales de los datos
            r = ultimo.get('tasa_real', 0) or 0
            g = ultimo.get('crecimiento_pib', 0) or 0
            balance_fiscal = ultimo.get('balance_fiscal_pib', 0) or 0
            deuda_pib_actual = ultimo['deuda_pib']
            deuda_pib_anterior = ultimo.get('deuda_pib_anterior', 0) or 0
            r_menos_g = r - g

            # Ecuación cambia según tipo de modelo
            if tipo_modelo == 'series_tiempo':
                ecuacion_html = f"""
                <div class="equation-box">
                    <strong>{ecuacion['nombre']}</strong><br>
                    <span style="font-size: 1rem; font-family: 'Courier New', monospace;">
                        {ecuacion['formula']}
                    </span>
                    <br><br>
                    <small>
                        <strong>Descripción:</strong> {ecuacion['descripcion']}<br><br>
                        ARIMA(p,d,q): yₜ = c + Σφᵢyₜ₋ᵢ + Σθⱼεₜ₋ⱼ + εₜ<br>
                        ETS: Sₜ = α·yₜ + (1-α)·(Sₜ₋₁ + Tₜ₋₁)
                    </small>
                </div>
                """
            elif tipo_modelo == 'machine_learning':
                ecuacion_html = f"""
                <div class="equation-box">
                    <strong>{ecuacion['nombre']}</strong><br>
                    <span style="font-size: 1rem; font-family: 'Courier New', monospace;">
                        {ecuacion['formula']}
                    </span>
                    <br><br>
                    <small>
                        <strong>Descripción:</strong> {ecuacion['descripcion']}<br><br>
                        Features: Inflación, Tasas MXN/USD, Tipo de Cambio, Lags(1,2,4)<br>
                        Linear: ŷ = β₀ + Σβᵢxᵢ | RF/GB: Ensemble de árboles
                    </small>
                </div>
                """
            else:  # hibrido
                ecuacion_html = f"""
                <div class="equation-box">
                    <strong>{ecuacion['nombre']}</strong><br>
                    <span style="font-size: 1rem; font-family: 'Courier New', monospace;">
                        {ecuacion['formula']}
                    </span>
                    <br><br>
                    <small>
                        <strong>Descripción:</strong> {ecuacion['descripcion']}<br><br>
                        ST: Ensemble(ARIMA, SARIMA, ETS)<br>
                        ML: Gradient Boosting con features macro
                    </small>
                </div>
                """

            st.markdown(ecuacion_html, unsafe_allow_html=True)

            with st.expander("Ver detalles del modelo y variables"):
                # Contenido según tipo de modelo
                if tipo_modelo == 'series_tiempo':
                    modelos_info = "ARIMA, SARIMA, ETS"
                    metodo_info = "Ponderación por RMSE inverso"
                elif tipo_modelo == 'machine_learning':
                    modelos_info = "Linear, Ridge, Random Forest, Gradient Boosting"
                    metodo_info = "Selección automática por menor RMSE"
                else:
                    modelos_info = "Series de Tiempo + Gradient Boosting"
                    metodo_info = "Combinación 50% ST + 50% ML"

                # Información del modelo
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.markdown("**Tipo de modelo**")
                    st.markdown(f"{tipo_modelo_label}")
                    st.markdown("**Modelos utilizados**")
                    st.markdown(f"{modelos_info}")
                with col_info2:
                    st.markdown("**Método de ensemble**")
                    st.markdown(f"{metodo_info}")

                st.markdown("---")
                st.markdown("**Variables macroeconómicas actuales**")

                # Tabla de variables con encabezados en bold
                interp_r_g = 'Deuda crece más rápido' if r_menos_g > 0 else 'Economía crece más rápido'
                st.markdown(f"""
                <table style="width: 100%; border-collapse: collapse; font-size: 0.9rem;">
                    <tr style="background: #F0F0F0;">
                        <th style="padding: 8px; text-align: left; border-bottom: 2px solid #D0E0EB; font-weight: 700;"></th>
                        <th style="padding: 8px; text-align: left; border-bottom: 2px solid #D0E0EB; font-weight: 700;">Variable</th>
                        <th style="padding: 8px; text-align: left; border-bottom: 2px solid #D0E0EB; font-weight: 700;">Valor</th>
                        <th style="padding: 8px; text-align: left; border-bottom: 2px solid #D0E0EB; font-weight: 700;">Interpretación</th>
                    </tr>
                    <tr style="background: #FFFFFF;">
                        <td style="padding: 6px 8px; border-bottom: 1px solid #E0E0E0;">0</td>
                        <td style="padding: 6px 8px; border-bottom: 1px solid #E0E0E0;">Deuda/PIB</td>
                        <td style="padding: 6px 8px; border-bottom: 1px solid #E0E0E0;">{deuda_pib_actual:.2f}%</td>
                        <td style="padding: 6px 8px; border-bottom: 1px solid #E0E0E0;">Ratio período actual</td>
                    </tr>
                    <tr style="background: #FFFFFF;">
                        <td style="padding: 6px 8px; border-bottom: 1px solid #E0E0E0;">1</td>
                        <td style="padding: 6px 8px; border-bottom: 1px solid #E0E0E0;">r (tasa real)</td>
                        <td style="padding: 6px 8px; border-bottom: 1px solid #E0E0E0;">{r:.2f}%</td>
                        <td style="padding: 6px 8px; border-bottom: 1px solid #E0E0E0;">Tasa nominal - Inflación</td>
                    </tr>
                    <tr style="background: #FFFFFF;">
                        <td style="padding: 6px 8px; border-bottom: 1px solid #E0E0E0;">2</td>
                        <td style="padding: 6px 8px; border-bottom: 1px solid #E0E0E0;">g (crecimiento PIB)</td>
                        <td style="padding: 6px 8px; border-bottom: 1px solid #E0E0E0;">{g:.2f}%</td>
                        <td style="padding: 6px 8px; border-bottom: 1px solid #E0E0E0;">Variación trimestral</td>
                    </tr>
                    <tr style="background: #FFFFFF;">
                        <td style="padding: 6px 8px;">3</td>
                        <td style="padding: 6px 8px;">r - g</td>
                        <td style="padding: 6px 8px;">{r_menos_g:.2f}%</td>
                        <td style="padding: 6px 8px;">{interp_r_g}</td>
                    </tr>
                </table>
                """, unsafe_allow_html=True)

        # KPI: Tendencia del indicador
        tendencia_deuda = procesador.calcular_tendencia(datos['df_analisis']['deuda_pib_ratio'])
        tendencia_texto = "Sube" if tendencia_deuda > 0 else "Baja"
        tendencia_color = COLORES['negativo'] if tendencia_deuda > 0 else COLORES['positivo']

        with col_kpi1:
            st.markdown("### 📈 KPI Tendencia")
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-value" style="color: {tendencia_color};">{tendencia_texto}</div>
                <div class="kpi-label">Indicador Deuda/PIB</div>
                <span style="color: {tendencia_color}; font-size: 0.9rem;">{tendencia_deuda:.4f}% por trimestre</span>
            </div>
            """, unsafe_allow_html=True)

        # KPI: Precisión del modelo (varía según tipo de modelo)
        if datos['metricas'] is not None:
            # Buscar métricas según tipo de modelo
            metricas_deuda = datos['metricas'][datos['metricas']['serie'].str.contains('Deuda_PIB', na=False)]
            if len(metricas_deuda) > 0:
                mejor_mape = metricas_deuda['MAPE'].min()
                mejor_modelo_nombre = metricas_deuda.loc[metricas_deuda['MAPE'].idxmin(), 'modelo']
            else:
                mejor_mape = 6.0
                mejor_modelo_nombre = 'N/A'
            precision = 100 - mejor_mape
        else:
            precision = 94.0
            mejor_modelo_nombre = 'N/A'

        with col_kpi2:
            st.markdown("### 🎯 KPI Precisión")
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-value" style="color: #27AE60;">{precision:.1f}%</div>
                <div class="kpi-label">Precisión ({tipo_modelo_label})</div>
                <span style="color: #6B645B; font-size: 0.9rem;">Mejor: {mejor_modelo_nombre} | MAPE: {100-precision:.2f}%</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # =============================================================================
        # FILA PRINCIPAL: Gráfico + Indicadores (MODO PREDICCIÓN)
        # =============================================================================
        col_main, col_indicators = st.columns([3, 1])

        with col_main:
            st.markdown(f"### 📊 Pronóstico Deuda/PIB ({tipo_modelo_label})")

            # Preparar datos del gráfico
            df_historico = datos['df_analisis'][['deuda_pib_ratio']].copy()
            df_pronostico = datos['pronosticos_deuda'][['Ensemble', 'Ensemble_lower', 'Ensemble_upper']].copy()

            # Filtrar por año/trimestre
            df_historico = filtrar_datos(df_historico, año_seleccionado, trimestre_seleccionado)

            # Si se filtró por año específico, también filtrar pronósticos
            if año_seleccionado != 'Todos':
                df_pronostico = filtrar_datos(df_pronostico, año_seleccionado, trimestre_seleccionado)

            # Crear gráfico
            fig = go.Figure()

            # Serie histórica (solo si hay datos)
            if not df_historico.empty:
                fig.add_trace(go.Scatter(
                    x=df_historico.index,
                    y=df_historico['deuda_pib_ratio'],
                    mode='lines+markers',
                    name='Histórico',
                    line=dict(color=COLORES['primario'], width=2),
                    marker=dict(size=6)
                ))

            # Pronóstico (solo si hay datos)
            if not df_pronostico.empty:
                fig.add_trace(go.Scatter(
                    x=df_pronostico.index,
                    y=df_pronostico['Ensemble'],
                    mode='lines+markers',
                    name=f'Pronóstico ({tipo_modelo_label})',
                    line=dict(color=COLORES['secundario'], width=2, dash='dash'),
                    marker=dict(size=6, symbol='diamond')
                ))

                # Intervalo de confianza
                fig.add_trace(go.Scatter(
                    x=list(df_pronostico.index) + list(df_pronostico.index)[::-1],
                    y=list(df_pronostico['Ensemble_upper']) + list(df_pronostico['Ensemble_lower'])[::-1],
                    fill='toself',
                    fillcolor='rgba(212, 175, 55, 0.4)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo='skip',
                    name='IC 95%'
                ))

            fig.update_layout(
                xaxis_title='Período',
                yaxis_title='Ratio Deuda/PIB (%)',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(color=COLORES['texto'])),
                plot_bgcolor=COLORES['fondo_card'],
                paper_bgcolor='rgba(0,0,0,0)',
                hovermode='x unified',
                margin=dict(l=50, r=50, t=30, b=50),
                font=dict(color=COLORES['texto'])
            )

            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.08)', title_font=dict(color=COLORES['texto'], size=14, family='Arial Black'), tickfont=dict(color=COLORES['texto']))
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.08)', title_font=dict(color=COLORES['texto'], size=14, family='Arial Black'), tickfont=dict(color=COLORES['texto']))

            st.plotly_chart(fig, use_container_width=True)

        with col_indicators:
            st.markdown("### 📌 MacroIndicadores")

            # Indicadores macroeconómicos actuales
            st.markdown(crear_indicador_macro(
                "PIB (USD 2018)",
                formato_numero(ultimo['pib'], 1, ' USD')
            ), unsafe_allow_html=True)

            st.markdown(crear_indicador_macro(
                "Deuda Pública",
                formato_numero(ultimo['deuda'], 1, ' USD')
            ), unsafe_allow_html=True)

            st.markdown(crear_indicador_macro(
                "Inflación YoY",
                f"{ultimo['inflacion']:.2f}%"
            ), unsafe_allow_html=True)

            st.markdown(crear_indicador_macro(
                "Tasa MXN",
                f"{ultimo['tasa_mxn']:.2f}%"
            ), unsafe_allow_html=True)

            st.markdown(crear_indicador_macro(
                "Tasa USD",
                f"{ultimo['tasa_usd']:.2f}%"
            ), unsafe_allow_html=True)

            st.markdown(crear_indicador_macro(
                "Tipo de Cambio",
                f"{ultimo['tipo_cambio']:.2f} MXN/USD"
            ), unsafe_allow_html=True)

        st.markdown("---")

        # =============================================================================
        # FILA INFERIOR: Tabla + FSI Pilares (MODO PREDICCIÓN)
        # =============================================================================
        col_tabla, col_fsi = st.columns([2, 1])

        with col_tabla:
            st.markdown("### 📋 Datos y Predicciones")

            # Preparar tabla combinada con filtros
            df_tabla_hist_base = datos['df_analisis'][['deuda_pib_ratio']].copy()
            df_tabla_pred_base = datos['pronosticos_deuda'][['Ensemble']].copy()

            # Aplicar filtros
            df_tabla_hist_filtrado = filtrar_datos(df_tabla_hist_base, año_seleccionado, trimestre_seleccionado)
            if año_seleccionado != 'Todos':
                df_tabla_pred_filtrado = filtrar_datos(df_tabla_pred_base, año_seleccionado, trimestre_seleccionado)
            else:
                df_tabla_pred_filtrado = df_tabla_pred_base

            # Tomar últimos 8 registros históricos (después de filtrar)
            df_tabla_hist = df_tabla_hist_filtrado.tail(8).copy()
            df_tabla_hist.columns = ['Deuda/PIB (%)']
            df_tabla_hist['Tipo'] = 'Histórico'
            df_tabla_hist['Predicción'] = '-'

            df_tabla_pred = df_tabla_pred_filtrado.copy()
            df_tabla_pred.columns = ['Deuda/PIB (%)']
            df_tabla_pred['Tipo'] = 'Pronóstico'
            df_tabla_pred['Predicción'] = df_tabla_pred['Deuda/PIB (%)'].apply(lambda x: f'{x:.2f}%')

            # Formatear para mostrar
            dfs_to_concat = []
            if not df_tabla_hist.empty:
                dfs_to_concat.append(df_tabla_hist)
            if not df_tabla_pred.empty:
                dfs_to_concat.append(df_tabla_pred)

            if dfs_to_concat:
                df_display = pd.concat(dfs_to_concat)
                df_display['Período'] = df_display.index.strftime('%Y-%m')
                df_display['Deuda/PIB (%)'] = df_display['Deuda/PIB (%)'].apply(lambda x: f'{x:.2f}%')
                df_display = df_display[['Período', 'Deuda/PIB (%)', 'Tipo', 'Predicción']]
                df_display = df_display.reset_index(drop=True)

                st.table(df_display)
            else:
                st.info("No hay datos para mostrar con los filtros seleccionados.")

        with col_fsi:
            st.markdown("### 🏛️ Pilares del FSI")

            # Gráfico de barras horizontales para pilares FSI
            pilares_nombres = ['Solvencia', 'Externa', 'Mercado', 'Liquidez', 'Fiscal']
            pilares_valores = [
                ultimo['p_solvencia'] or 0,
                ultimo['p_externa'] or 0,
                ultimo['p_mercado'] or 0,
                ultimo['p_liquidez'] or 0,
                ultimo['p_fiscal'] or 0
            ]

            colores_pilares = [
                COLORES['positivo'] if v >= 50 else COLORES['negativo']
                for v in pilares_valores
            ]

            fig_fsi = go.Figure(go.Bar(
                y=pilares_nombres,
                x=pilares_valores,
                orientation='h',
                marker_color=colores_pilares,
                text=[f'{v:.1f}' for v in pilares_valores],
                textposition='outside'
            ))

            fig_fsi.add_vline(x=50, line_dash='dash', line_color='orange', annotation_text='Neutral')

            fig_fsi.update_layout(
                xaxis_title='Score (0-100)',
                yaxis_title='',
                plot_bgcolor=COLORES['fondo_card'],
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=50, t=10, b=30),
                xaxis=dict(range=[0, 100]),
                font=dict(color=COLORES['texto'])
            )

            fig_fsi.update_xaxes(title_font=dict(color=COLORES['texto'], size=14, family='Arial Black'), tickfont=dict(color=COLORES['texto']))
            fig_fsi.update_yaxes(title_font=dict(color=COLORES['texto'], size=14, family='Arial Black'), tickfont=dict(color=COLORES['texto'], size=12, family='Arial Black'))

            st.plotly_chart(fig_fsi, use_container_width=True)

            # Estado general del FSI
            fsi_actual = ultimo['fsi']
            riesgo_fsi, color_fsi = procesador.clasificar_riesgo_fsi(fsi_actual)

            st.markdown(f"""
            <div class="status-pill" style="background-color: {color_fsi}; margin-top: 10px;">
                <strong style="color: #000000 !important; font-weight: 700;">FSI Actual: {fsi_actual:.1f}</strong><br>
                <span style="color: #000000 !important; font-weight: 400;">Estado: {riesgo_fsi}</span>
            </div>
            """, unsafe_allow_html=True)

            # Identificar pilares críticos y fuertes
            pilares_dict = {
                'Solvencia': ultimo['p_solvencia'] or 0,
                'Externa': ultimo['p_externa'] or 0,
                'Mercado': ultimo['p_mercado'] or 0,
                'Liquidez': ultimo['p_liquidez'] or 0,
                'Fiscal': ultimo['p_fiscal'] or 0
            }
            pilares_criticos = [p for p, v in pilares_dict.items() if v < 40]
            pilares_debiles = [p for p, v in pilares_dict.items() if 40 <= v < 50]
            pilares_fuertes = [p for p, v in pilares_dict.items() if v >= 70]

            with st.expander("📊 Ver razones del estado FSI"):
                st.markdown(f"**Estado actual: {riesgo_fsi}** (FSI = {fsi_actual:.1f})")

                if pilares_criticos:
                    st.markdown(f"🔴 **Pilares críticos (<40):** {', '.join(pilares_criticos)}")
                if pilares_debiles:
                    st.markdown(f"🟠 **Pilares débiles (40-50):** {', '.join(pilares_debiles)}")
                if pilares_fuertes:
                    st.markdown(f"🟢 **Pilares fuertes (>70):** {', '.join(pilares_fuertes)}")

                st.markdown("---")
                st.markdown("**Detalle por pilar:**")
                for pilar, valor in pilares_dict.items():
                    if valor < 40:
                        emoji = "🔴"
                        estado = "Crítico"
                    elif valor < 50:
                        emoji = "🟠"
                        estado = "Débil"
                    elif valor < 70:
                        emoji = "🟡"
                        estado = "Moderado"
                    else:
                        emoji = "🟢"
                        estado = "Fuerte"
                    st.markdown(f"{emoji} **{pilar}:** {valor:.1f} - {estado}")

        # =============================================================================
        # GRÁFICO ADICIONAL: FSI con Predicción (solo en modo predicción)
        # =============================================================================
        st.markdown("---")
        st.markdown("### 📈 Índice de Sostenibilidad Financiera (FSI) con Predicción")

        col_fsi_graph, col_fsi_info = st.columns([3, 1])

        with col_fsi_graph:
            # Gráfico FSI con filtros aplicados
            df_fsi_hist = datos['df_analisis'][['FSI']].dropna()
            df_fsi_pred = datos['pronosticos_fsi'][['Ensemble', 'Ensemble_lower', 'Ensemble_upper']]

            # Aplicar filtros
            df_fsi_hist = filtrar_datos(df_fsi_hist, año_seleccionado, trimestre_seleccionado)
            if año_seleccionado != 'Todos':
                df_fsi_pred = filtrar_datos(df_fsi_pred, año_seleccionado, trimestre_seleccionado)

            fig_fsi_full = go.Figure()

            # Histórico (solo si hay datos)
            if not df_fsi_hist.empty:
                fig_fsi_full.add_trace(go.Scatter(
                    x=df_fsi_hist.index,
                    y=df_fsi_hist['FSI'],
                    mode='lines+markers',
                    name='FSI Histórico',
                    line=dict(color='#27AE60', width=2),
                    marker=dict(size=5)
                ))

            # Pronóstico (solo si hay datos)
            if not df_fsi_pred.empty:
                fig_fsi_full.add_trace(go.Scatter(
                    x=df_fsi_pred.index,
                    y=df_fsi_pred['Ensemble'],
                    mode='lines+markers',
                    name='FSI Pronóstico',
                    line=dict(color='#E74C3C', width=2, dash='dash'),
                    marker=dict(size=5, symbol='diamond')
                ))

                # Intervalo
                fig_fsi_full.add_trace(go.Scatter(
                    x=list(df_fsi_pred.index) + list(df_fsi_pred.index)[::-1],
                    y=list(df_fsi_pred['Ensemble_upper']) + list(df_fsi_pred['Ensemble_lower'])[::-1],
                    fill='toself',
                    fillcolor='rgba(231, 76, 60, 0.4)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo='skip',
                    name='IC 95%'
                ))

            # Línea neutral
            fig_fsi_full.add_hline(y=50, line_dash='dot', line_color='orange',
                                   annotation_text='Neutral (50)')

            fig_fsi_full.update_layout(
                xaxis_title='Período',
                yaxis_title='FSI (0-100)',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(color=COLORES['texto'])),
                plot_bgcolor=COLORES['fondo_card'],
                paper_bgcolor='rgba(0,0,0,0)',
                hovermode='x unified',
                margin=dict(l=50, r=50, t=30, b=50),
                font=dict(color=COLORES['texto'])
            )

            fig_fsi_full.update_xaxes(title_font=dict(color=COLORES['texto'], size=14, family='Arial Black'), tickfont=dict(color=COLORES['texto']))
            fig_fsi_full.update_yaxes(title_font=dict(color=COLORES['texto'], size=14, family='Arial Black'), tickfont=dict(color=COLORES['texto']))

            st.plotly_chart(fig_fsi_full, use_container_width=True)

        with col_fsi_info:
            st.markdown("### 📊 Resumen FSI")

            # Métricas FSI
            fsi_actual = ultimo['fsi']
            st.metric("FSI Actual", f"{fsi_actual:.1f}",
                      delta=f"{procesador.calcular_tendencia(datos['df_analisis']['FSI']):.2f}")

            # Pronósticos (con validación)
            pronosticos_fsi = datos['pronosticos_fsi']
            if len(pronosticos_fsi) > 3:
                st.metric("Pronóstico 1 año", f"{pronosticos_fsi['Ensemble'].iloc[3]:.1f}")
            if len(pronosticos_fsi) > 0:
                st.metric("Pronóstico 3 años", f"{pronosticos_fsi['Ensemble'].iloc[-1]:.1f}")

            # Clasificación de riesgo deuda
            deuda_actual = ultimo['deuda_pib']
            riesgo_deuda, color_deuda = procesador.clasificar_riesgo_deuda(deuda_actual)

            st.markdown(f"""
            <div class="status-pill" style="background-color: {color_deuda}; margin-top: 20px;">
                <strong style="color: #000000 !important; font-weight: 700;">Deuda/PIB: {deuda_actual:.1f}%</strong><br>
                <span style="color: #000000 !important; font-weight: 400;">Riesgo: {riesgo_deuda}</span>
            </div>
            """, unsafe_allow_html=True)

            # Calcular factores que influyen en el estado de deuda
            tasa_real = ultimo['tasa_mxn'] - ultimo['inflacion'] if ultimo['tasa_mxn'] and ultimo['inflacion'] else 0

            with st.expander("📈 Ver razones del estado Deuda/PIB"):
                st.markdown(f"**Nivel de riesgo: {riesgo_deuda}** (Deuda/PIB = {deuda_actual:.1f}%)")

                st.markdown("---")
                st.markdown("**Factores macroeconómicos actuales:**")

                # Tasa de interés real
                if tasa_real > 3:
                    st.markdown(f"🔴 **Tasa de interés real:** {tasa_real:.2f}% (alta - presiona al alza la deuda)")
                elif tasa_real > 0:
                    st.markdown(f"🟡 **Tasa de interés real:** {tasa_real:.2f}% (positiva)")
                else:
                    st.markdown(f"🟢 **Tasa de interés real:** {tasa_real:.2f}% (negativa - favorable)")

                # Inflación
                if ultimo['inflacion'] > 6:
                    st.markdown(f"🔴 **Inflación:** {ultimo['inflacion']:.2f}% (alta)")
                elif ultimo['inflacion'] > 4:
                    st.markdown(f"🟡 **Inflación:** {ultimo['inflacion']:.2f}% (moderada)")
                else:
                    st.markdown(f"🟢 **Inflación:** {ultimo['inflacion']:.2f}% (controlada)")

                # Tipo de cambio
                if ultimo['tipo_cambio'] > 20:
                    st.markdown(f"🟠 **Tipo de cambio:** {ultimo['tipo_cambio']:.2f} MXN/USD (peso débil)")
                else:
                    st.markdown(f"🟢 **Tipo de cambio:** {ultimo['tipo_cambio']:.2f} MXN/USD")

                # Tendencia
                tendencia = procesador.calcular_tendencia(datos['df_analisis']['deuda_pib_ratio'])
                if tendencia > 0.5:
                    st.markdown(f"🔴 **Tendencia:** +{tendencia:.2f}% por trimestre (creciendo rápidamente)")
                elif tendencia > 0:
                    st.markdown(f"🟡 **Tendencia:** +{tendencia:.2f}% por trimestre (creciendo)")
                else:
                    st.markdown(f"🟢 **Tendencia:** {tendencia:.2f}% por trimestre (decreciendo)")

    else:
        # =============================================================================
        # LAYOUT MODO COMPARATIVA (completamente separado del modo predicción)
        # =============================================================================
        if df_deuda_banxico is not None and df_pib_bm is not None:
            st.markdown("---")
            st.markdown("### 📊 Comparativa Corriente: Modelo vs Fuentes Externas")

            st.info("""
            **Nota sobre la comparación:** Ambas fuentes utilizan **PIB Corriente en USD** y **Deuda Pública Federal**.
            Se compara el ratio Deuda/PIB del modelo (SHCP/INEGI) con el calculado usando datos de Banxico y Banco Mundial.
            """)

            # Preparar serie del modelo
            serie_modelo = datos['df_analisis']['deuda_pib_ratio'].copy()
            serie_modelo.index = pd.DatetimeIndex(serie_modelo.index).to_period('Q').to_timestamp()

            # Preparar serie de Banxico (calcular ratio Deuda/PIB)
            # Primero necesitamos alinear PIB anual con datos trimestrales
            df_deuda_banxico_calc = df_deuda_banxico.copy()
            df_deuda_banxico_calc.index = pd.DatetimeIndex(df_deuda_banxico_calc.index).to_period('Q').to_timestamp()

            # Crear serie de PIB trimestral a partir de datos anuales del Banco Mundial
            df_pib_bm_trimestral = pd.DataFrame()
            for _, row in df_pib_bm.iterrows():
                anio = int(row['Anio'])
                pib_musd = row['PIB_MUSD']
                # Crear 4 trimestres con el mismo PIB anual
                for q in range(1, 5):
                    fecha = pd.Timestamp(f'{anio}-{(q-1)*3+1:02d}-01')
                    df_pib_bm_trimestral = pd.concat([
                        df_pib_bm_trimestral,
                        pd.DataFrame({'PIB_BM_MUSD': [pib_musd]}, index=[fecha])
                    ])

            df_pib_bm_trimestral.index = pd.DatetimeIndex(df_pib_bm_trimestral.index).to_period('Q').to_timestamp()

            # Calcular ratio Deuda/PIB usando datos de Banxico (Deuda Pública Federal) y Banco Mundial
            idx_comun = df_deuda_banxico_calc.index.intersection(df_pib_bm_trimestral.index)
            df_comparacion = pd.DataFrame(index=idx_comun)
            df_comparacion['Deuda_Federal_Banxico_MUSD'] = df_deuda_banxico_calc.loc[idx_comun, 'Deuda_Publica_Federal_MUSD']
            df_comparacion['PIB_BM_MUSD'] = df_pib_bm_trimestral.loc[idx_comun, 'PIB_BM_MUSD']
            df_comparacion['Deuda_PIB_Externo'] = (df_comparacion['Deuda_Federal_Banxico_MUSD'] / df_comparacion['PIB_BM_MUSD']) * 100

            serie_externa = df_comparacion['Deuda_PIB_Externo'].dropna()

            # Calcular métricas de similitud
            metricas = calcular_metricas_similitud(serie_modelo, serie_externa, 'Modelo vs Banxico/BM')

            col_comp1, col_comp2 = st.columns([2, 1])

            with col_comp1:
                # Gráfico comparativo
                fig_comp = go.Figure()

                # Serie del modelo
                idx_plot = serie_modelo.index.intersection(serie_externa.index)
                fig_comp.add_trace(go.Scatter(
                    x=serie_modelo.loc[idx_plot].index,
                    y=serie_modelo.loc[idx_plot].values,
                    mode='lines+markers',
                    name='Modelo (SHCP/INEGI)',
                    line=dict(color=COLORES['primario'], width=2),
                    marker=dict(size=6)
                ))

                # Serie externa
                fig_comp.add_trace(go.Scatter(
                    x=serie_externa.loc[idx_plot].index,
                    y=serie_externa.loc[idx_plot].values,
                    mode='lines+markers',
                    name='Externo (Banxico/BM)',
                    line=dict(color=COLORES['secundario'], width=2),
                    marker=dict(size=6)
                ))

                fig_comp.update_layout(
                    xaxis_title='Período',
                    yaxis_title='Ratio Deuda/PIB (%)',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(color=COLORES['texto'])),
                    plot_bgcolor=COLORES['fondo_card'],
                    paper_bgcolor='rgba(0,0,0,0)',
                    hovermode='x unified',
                    margin=dict(l=50, r=50, t=30, b=50),
                    font=dict(color=COLORES['texto'])
                )

                fig_comp.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.08)', title_font=dict(color=COLORES['texto'], size=14, family='Arial Black'), tickfont=dict(color=COLORES['texto']))
                fig_comp.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.08)', title_font=dict(color=COLORES['texto'], size=14, family='Arial Black'), tickfont=dict(color=COLORES['texto']))

                st.plotly_chart(fig_comp, use_container_width=True)

            with col_comp2:
                st.markdown("### 📈 Métricas de Similitud")

                if metricas:
                    # Clasificar similitud
                    if metricas['correlacion'] >= 0.95:
                        color_corr = COLORES['positivo']
                        estado_corr = "Muy Alta"
                    elif metricas['correlacion'] >= 0.85:
                        color_corr = '#f39c12'
                        estado_corr = "Alta"
                    elif metricas['correlacion'] >= 0.70:
                        color_corr = COLORES['neutro']
                        estado_corr = "Moderada"
                    else:
                        color_corr = COLORES['negativo']
                        estado_corr = "Baja"

                    st.markdown(f"""
                    <div class="kpi-box">
                        <div class="kpi-value" style="color: {color_corr};">{metricas['correlacion']:.4f}</div>
                        <div class="kpi-label">Correlación</div>
                        <span style="color: {color_corr}; font-size: 0.9rem;">{estado_corr}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class="macro-indicator">
                        <strong>RMSE</strong><br>
                        <span style="font-size: 1.25rem; color: {COLORES['texto']}; font-weight: 400;">{metricas['rmse']:.2f}%</span>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class="macro-indicator">
                        <strong>MAPE</strong><br>
                        <span style="font-size: 1.25rem; color: {COLORES['texto']}; font-weight: 400;">{metricas['mape']:.2f}%</span>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class="macro-indicator">
                        <strong>Observaciones</strong><br>
                        <span style="font-size: 1.25rem; color: {COLORES['texto']}; font-weight: 400;">{metricas['n_observaciones']}</span>
                    </div>
                    """, unsafe_allow_html=True)

            # Análisis detallado expandible
            with st.expander("📊 Ver análisis detallado de la comparación"):
                if metricas:
                    st.markdown(f"""
                    **Período de comparación:** {metricas['n_observaciones']} trimestres

                    **Interpretación de métricas:**

                    | Métrica | Valor | Interpretación |
                    |---------|-------|----------------|
                    | **Correlación** | {metricas['correlacion']:.4f} | {'Las series siguen patrones muy similares' if metricas['correlacion'] > 0.9 else 'Las series tienen patrones moderadamente similares'} |
                    | **RMSE** | {metricas['rmse']:.2f}% | Error cuadrático medio entre las series |
                    | **MAPE** | {metricas['mape']:.2f}% | Error porcentual absoluto medio |
                    | **MAE** | {metricas['mae']:.2f}% | Diferencia absoluta promedio |
                    | **Diff. Promedio** | {metricas['diff_promedio']:+.2f}% | {'Serie externa mayor' if metricas['diff_promedio'] > 0 else 'Serie del modelo mayor'} |
                    """)

                    st.markdown("---")
                    st.markdown("**Fuentes de datos:**")
                    st.markdown("""
                    - **Modelo (SHCP/INEGI):** Deuda Pública Federal y PIB Corriente procesados
                    - **Externo (Banxico/BM):** Deuda Pública Federal (Banxico) + PIB Corriente (Banco Mundial)
                    """)

                    # Gráfico de diferencias
                    st.markdown("---")
                    st.markdown("**Diferencia entre series (Externo - Modelo):**")

                    diferencias = serie_externa.loc[idx_plot] - serie_modelo.loc[idx_plot]
                    fig_diff = go.Figure()
                    fig_diff.add_trace(go.Bar(
                        x=diferencias.index,
                        y=diferencias.values,
                        marker_color=[COLORES['positivo'] if v >= 0 else COLORES['negativo'] for v in diferencias.values],
                        name='Diferencia'
                    ))
                    fig_diff.add_hline(y=0, line_dash='dash', line_color='gray')
                    fig_diff.update_layout(
                        xaxis_title='Período',
                        yaxis_title='Diferencia (p.p.)',
                        plot_bgcolor=COLORES['fondo_card'],
                        paper_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=50, r=50, t=30, b=50),
                        font=dict(color=COLORES['texto']),
                        height=300
                    )
                    fig_diff.update_xaxes(title_font=dict(color=COLORES['texto'], size=14, family='Arial Black'), tickfont=dict(color=COLORES['texto']))
                    fig_diff.update_yaxes(title_font=dict(color=COLORES['texto'], size=14, family='Arial Black'), tickfont=dict(color=COLORES['texto']))
                    st.plotly_chart(fig_diff, use_container_width=True)
                else:
                    st.warning("No se pudieron calcular métricas. Verifica que los datos secundarios estén disponibles.")
        else:
            st.error("""
            **No se pudieron cargar los datos secundarios de Banxico/Banco Mundial para la comparación.**

            Para habilitar el modo comparativa, ejecute el notebook `ETL_Secundarias.ipynb` que descarga
            y procesa los datos de:
            - PIB Corriente del Banco Mundial
            - Deuda Pública Federal de Banxico

            Los archivos se guardarán en: `Datos_Resultado/DatasetSecundarioFinal/`
            """)

    # =============================================================================
    # FOOTER
    # =============================================================================
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6B645B; padding: 20px;">
        <small>Dashboard desarrollado para análisis de sostenibilidad fiscal de México</small><br>
        <small>Datos: SHCP, Banxico, INEGI | Modelos: ARIMA, SARIMA, ETS</small><br><br>
        <em style="font-style: italic;">"La técnica al servicio de la patria"</em>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
