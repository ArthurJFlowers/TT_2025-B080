# -*- coding: utf-8 -*-
"""
Procesamiento de Datos - Dashboard Deuda Externa México
Basado en PIB_DEUDA_EXTERNA.ipynb, Modelo.ipynb y notebooks de Evaluación 8 Pilares
Soporta modelos: Machine Learning, Series de Tiempo e Híbrido
"""

import os
import warnings
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Suprimir warnings
warnings.filterwarnings('ignore')
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)

# Modelos de series de tiempo
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Modelos de Machine Learning
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

try:
    import pmdarima as pm
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False


class ProcesadorDatos:
    """Clase para procesar y preparar datos del dashboard"""

    # Tipos de proceso disponibles
    PROCESOS_DISPONIBLES = ['base_2018', 'cuenta_corriente']

    # Tipos de modelo disponibles
    MODELOS_DISPONIBLES = ['series_tiempo', 'machine_learning', 'hibrido']

    def __init__(self, base_path=None, proceso='base_2018', tipo_modelo='series_tiempo'):
        if base_path is None:
            base_path = Path(__file__).parent
        self.base_path = Path(base_path)
        self.PATH_DATASET = self.base_path / "Datos_Resultado" / "DatasetFinal"
        self.PATH_INDICADORES = self.PATH_DATASET / "Indicadores"
        self.PATH_SECUNDARIOS = self.base_path / "Datos_Resultado" / "DatasetSecundarioFinal"

        # Tipo de proceso seleccionado
        self.proceso = proceso if proceso in self.PROCESOS_DISPONIBLES else 'base_2018'

        # Tipo de modelo seleccionado
        self.tipo_modelo = tipo_modelo if tipo_modelo in self.MODELOS_DISPONIBLES else 'series_tiempo'

        # DataFrames
        self.df_main = None
        self.df_fsi = None
        self.df_components = None
        self.df_analisis = None
        self.pronosticos_deuda = None
        self.pronosticos_fsi = None
        self.metricas = None

        # Modelos ML entrenados
        self.modelos_ml = {}
        self.scaler = None

    def set_proceso(self, proceso):
        """Cambia el tipo de proceso y recarga datos"""
        if proceso in self.PROCESOS_DISPONIBLES:
            self.proceso = proceso
            return True
        return False

    def get_proceso(self):
        """Retorna el proceso actual"""
        return self.proceso

    def set_tipo_modelo(self, tipo_modelo):
        """Cambia el tipo de modelo (series_tiempo, machine_learning, hibrido)"""
        if tipo_modelo in self.MODELOS_DISPONIBLES:
            self.tipo_modelo = tipo_modelo
            return True
        return False

    def get_tipo_modelo(self):
        """Retorna el tipo de modelo actual"""
        return self.tipo_modelo

    def _get_archivo_dataset(self):
        """Retorna el nombre del archivo según el proceso"""
        if self.proceso == 'base_2018':
            return "DatasetFinal_USD_2018_reducido.csv"
        else:  # cuenta_corriente
            return "DatasetFinal_USD_Corrientes_reducido.csv"

    def _get_archivo_fsi(self):
        """Retorna el nombre del archivo FSI según el proceso"""
        if self.proceso == 'base_2018':
            return "fsi_timeseries_USD_2018.csv"
        else:
            return "fsi_timeseries_USD_Corrientes.csv"

    def _get_archivo_components(self):
        """Retorna el nombre del archivo de componentes según el proceso"""
        if self.proceso == 'base_2018':
            return "fsi_components_USD_2018.csv"
        else:
            return "fsi_components_USD_Corrientes.csv"

    def cargar_datos(self):
        """Carga todos los datasets necesarios según el proceso seleccionado"""
        try:
            archivo_dataset = self._get_archivo_dataset()
            archivo_fsi = self._get_archivo_fsi()
            archivo_components = self._get_archivo_components()

            # Dataset principal
            self.df_main = pd.read_csv(
                self.PATH_DATASET / archivo_dataset,
                parse_dates=['periodo']
            )
            self.df_main.set_index('periodo', inplace=True)

            # FSI Timeseries
            self.df_fsi = pd.read_csv(
                self.PATH_INDICADORES / archivo_fsi,
                parse_dates=['periodo']
            )
            self.df_fsi.set_index('periodo', inplace=True)

            # FSI Components
            self.df_components = pd.read_csv(
                self.PATH_INDICADORES / archivo_components,
                parse_dates=['periodo']
            )
            self.df_components.set_index('periodo', inplace=True)

            # Preparar serie de análisis
            self._preparar_series()

            print(f"Datos cargados correctamente (proceso: {self.proceso})")
            return True
        except Exception as e:
            print(f"Error cargando datos: {e}")
            return False

    def _preparar_series(self):
        """Prepara las series para análisis"""
        # Establecer frecuencia trimestral
        self.df_main.index = pd.DatetimeIndex(self.df_main.index).to_period('Q').to_timestamp()
        self.df_main = self.df_main[~self.df_main.index.duplicated(keep='first')]
        self.df_main = self.df_main.asfreq('QS')

        # Determinar columnas según proceso
        if self.proceso == 'base_2018':
            col_deuda = 'deuda_bruta_publica_musd_2018'
            col_pib = 'PIB_USD_2018'
        else:
            # Cuenta corriente - buscar nombres de columnas reales
            col_deuda = None
            col_pib = None
            for col in self.df_main.columns:
                if 'deuda_bruta_publica' in col.lower() and col_deuda is None:
                    col_deuda = col
                if 'pib_usd' in col.lower() and 'corriente' in col.lower() and col_pib is None:
                    col_pib = col
            # Fallback a base 2018 si no se encuentran
            if col_deuda is None:
                col_deuda = 'deuda_bruta_publica_musd_2018'
            if col_pib is None:
                col_pib = 'PIB_USD_2018'

        # Guardar nombres de columnas para uso posterior
        self.col_deuda = col_deuda
        self.col_pib = col_pib

        # Calcular ratio Deuda/PIB
        self.df_main['deuda_pib_ratio'] = (
            self.df_main[col_deuda] / self.df_main[col_pib]
        ) * 100

        # Preparar FSI
        self.df_fsi.index = pd.DatetimeIndex(self.df_fsi.index).to_period('Q').to_timestamp()
        self.df_fsi = self.df_fsi[~self.df_fsi.index.duplicated(keep='first')]
        self.df_fsi = self.df_fsi.asfreq('QS')

        # Combinar para análisis - incluir columnas detectadas dinámicamente
        # Incluir balance fiscal para la ecuación
        cols_base = [col_pib, col_deuda, 'deuda_pib_ratio', 'inflacion_yoy_avg',
                     'TasaMXN', 'TasaUSD', 'tcn_mean', 'balance_mmxn_USD_2018',
                     'balance_mmxn_USD_corriente', 'ingresos_totales_miles_USD_2018',
                     'gasto_totales_miles_USD_2018']
        cols_disponibles = [c for c in cols_base if c in self.df_main.columns]

        self.df_analisis = self.df_main[cols_disponibles].copy()

        # Calcular balance fiscal como % del PIB si hay datos
        if 'balance_mmxn_USD_2018' in self.df_analisis.columns:
            self.df_analisis['balance_fiscal_pib'] = (
                self.df_analisis['balance_mmxn_USD_2018'] / self.df_analisis[col_pib]
            ) * 100
        elif 'balance_mmxn_USD_corriente' in self.df_analisis.columns:
            self.df_analisis['balance_fiscal_pib'] = (
                self.df_analisis['balance_mmxn_USD_corriente'] / self.df_analisis[col_pib]
            ) * 100

        # Calcular crecimiento del PIB (g)
        self.df_analisis['crecimiento_pib'] = self.df_analisis[col_pib].pct_change() * 100

        # Calcular tasa de interés real (r)
        if 'TasaMXN' in self.df_analisis.columns and 'inflacion_yoy_avg' in self.df_analisis.columns:
            self.df_analisis['tasa_real'] = (
                self.df_analisis['TasaMXN'] - self.df_analisis['inflacion_yoy_avg']
            )

        # Unir FSI
        fsi_cols = ['FSI', 'p_solvencia', 'p_externa', 'p_mercado', 'p_liquidez', 'p_fiscal']
        fsi_disponibles = [col for col in fsi_cols if col in self.df_fsi.columns]
        self.df_analisis = self.df_analisis.join(self.df_fsi[fsi_disponibles])

    def generar_pronosticos(self, periodos=12):
        """Genera pronósticos según el tipo de modelo seleccionado"""
        print(f"Generando pronósticos (proceso: {self.proceso}, modelo: {self.tipo_modelo})...")

        if self.tipo_modelo == 'series_tiempo':
            return self._generar_pronosticos_series_tiempo(periodos)
        elif self.tipo_modelo == 'machine_learning':
            return self._generar_pronosticos_ml(periodos)
        elif self.tipo_modelo == 'hibrido':
            return self._generar_pronosticos_hibrido(periodos)
        else:
            return self._generar_pronosticos_series_tiempo(periodos)

    def _generar_pronosticos_series_tiempo(self, periodos=12):
        """Genera pronósticos usando modelos de series de tiempo (ARIMA, SARIMA, ETS)"""
        # Modelo para Deuda/PIB
        modelo_deuda = ModeloPrediccion(
            self.df_analisis['deuda_pib_ratio'],
            'Ratio Deuda/PIB (%)'
        )
        modelo_deuda.dividir_train_test(test_size=8)
        modelo_deuda.entrenar_todos()
        self.pronosticos_deuda = modelo_deuda.generar_pronostico(periodos=periodos)

        # Modelo para FSI
        modelo_fsi = ModeloPrediccion(
            self.df_analisis['FSI'].dropna(),
            'FSI (0-100)'
        )
        modelo_fsi.dividir_train_test(test_size=8)
        modelo_fsi.entrenar_todos()
        self.pronosticos_fsi = modelo_fsi.generar_pronostico(periodos=periodos)

        # Guardar métricas
        metricas_deuda = pd.DataFrame([
            modelo_deuda.resultados[m]['metricas']
            for m in modelo_deuda.resultados
        ])
        metricas_fsi = pd.DataFrame([
            modelo_fsi.resultados[m]['metricas']
            for m in modelo_fsi.resultados
        ])
        metricas_deuda['serie'] = 'Deuda_PIB'
        metricas_fsi['serie'] = 'FSI'
        self.metricas = pd.concat([metricas_deuda, metricas_fsi])

        return True

    def _generar_pronosticos_ml(self, periodos=12):
        """Genera pronósticos usando modelos de Machine Learning"""
        print("  Entrenando modelos de Machine Learning...")

        # Preparar features para ML
        df_ml = self.df_analisis.copy().dropna()

        # Features disponibles para predicción
        feature_cols = []
        for col in ['inflacion_yoy_avg', 'TasaMXN', 'TasaUSD', 'tcn_mean',
                    'crecimiento_pib', 'tasa_real', 'FSI']:
            if col in df_ml.columns and df_ml[col].notna().sum() > 10:
                feature_cols.append(col)

        # Agregar lags como features
        for lag in [1, 2, 4]:
            df_ml[f'deuda_pib_lag{lag}'] = df_ml['deuda_pib_ratio'].shift(lag)
            feature_cols.append(f'deuda_pib_lag{lag}')

        df_ml = df_ml.dropna()

        if len(df_ml) < 20:
            print("  Datos insuficientes para ML, usando series de tiempo")
            return self._generar_pronosticos_series_tiempo(periodos)

        X = df_ml[feature_cols]
        y = df_ml['deuda_pib_ratio']

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # Escalar features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Entrenar modelos
        modelos = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }

        resultados_ml = []
        for nombre, modelo in modelos.items():
            print(f"    Entrenando {nombre}...")
            if 'Forest' in nombre or 'Boosting' in nombre:
                modelo.fit(X_train, y_train)
                y_pred = modelo.predict(X_test)
            else:
                modelo.fit(X_train_scaled, y_train)
                y_pred = modelo.predict(X_test_scaled)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100

            resultados_ml.append({
                'modelo': nombre, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape
            })
            self.modelos_ml[nombre] = modelo

            print(f"      {nombre} - RMSE: {rmse:.4f}, R²: {1 - (rmse**2 / np.var(y_test)):.4f}")

        # Seleccionar mejor modelo
        mejor = min(resultados_ml, key=lambda x: x['RMSE'])
        print(f"  Mejor modelo ML: {mejor['modelo']} (RMSE: {mejor['RMSE']:.4f})")

        # Generar pronósticos con el mejor modelo
        self.pronosticos_deuda = self._pronosticar_ml(
            df_ml, feature_cols, periodos, mejor['modelo'], mejor['RMSE']
        )

        # Para FSI usar series de tiempo (ML no aplica bien)
        modelo_fsi = ModeloPrediccion(
            self.df_analisis['FSI'].dropna(),
            'FSI (0-100)'
        )
        modelo_fsi.dividir_train_test(test_size=8)
        modelo_fsi.entrenar_todos()
        self.pronosticos_fsi = modelo_fsi.generar_pronostico(periodos=periodos)

        # Guardar métricas
        metricas_deuda = pd.DataFrame(resultados_ml)
        metricas_deuda['serie'] = 'Deuda_PIB'
        metricas_fsi = pd.DataFrame([
            modelo_fsi.resultados[m]['metricas']
            for m in modelo_fsi.resultados
        ])
        metricas_fsi['serie'] = 'FSI'
        self.metricas = pd.concat([metricas_deuda, metricas_fsi])

        return True

    def _pronosticar_ml(self, df_ml, feature_cols, periodos, mejor_modelo, rmse_modelo=None):
        """Genera pronósticos futuros con el modelo ML seleccionado"""
        ultima_fecha = df_ml.index[-1]
        fechas_futuras = pd.date_range(
            start=ultima_fecha + pd.DateOffset(months=3),
            periods=periodos, freq='QS'
        )

        df_pron = pd.DataFrame(index=fechas_futuras)

        # Proyectar features hacia el futuro (usando últimos valores + tendencia)
        ultimo_row = df_ml.iloc[-1].copy()

        pronosticos = []
        for i in range(periodos):
            # Crear features para este período
            X_future = {}
            for col in feature_cols:
                if 'lag' in col:
                    lag_num = int(col.split('lag')[-1])
                    if i >= lag_num:
                        X_future[col] = pronosticos[i - lag_num]
                    else:
                        idx = -(lag_num - i)
                        X_future[col] = df_ml['deuda_pib_ratio'].iloc[idx]
                else:
                    # Usar último valor conocido con pequeña variación aleatoria
                    X_future[col] = ultimo_row[col]

            X_df = pd.DataFrame([X_future])[feature_cols]

            # Predecir
            modelo = self.modelos_ml[mejor_modelo]
            if 'Forest' in mejor_modelo or 'Boosting' in mejor_modelo:
                pred = modelo.predict(X_df)[0]
            else:
                X_scaled = self.scaler.transform(X_df)
                pred = modelo.predict(X_scaled)[0]

            pronosticos.append(pred)

        df_pron[mejor_modelo] = pronosticos
        df_pron['Ensemble'] = pronosticos

        # Intervalos de confianza basados en error histórico
        if rmse_modelo is not None:
            rmse_hist = rmse_modelo
        elif self.metricas is not None:
            rmse_hist = self.metricas[self.metricas['modelo'] == mejor_modelo]['RMSE'].values[0]
        else:
            # Valor por defecto si no hay métricas disponibles
            rmse_hist = 1.0
        df_pron['Ensemble_std'] = rmse_hist
        df_pron['Ensemble_lower'] = df_pron['Ensemble'] - 1.96 * rmse_hist
        df_pron['Ensemble_upper'] = df_pron['Ensemble'] + 1.96 * rmse_hist

        return df_pron

    def _generar_pronosticos_hibrido(self, periodos=12):
        """Genera pronósticos combinando Series de Tiempo y Machine Learning"""
        print("  Generando pronósticos híbridos (Series de Tiempo + ML)...")

        # Primero generar pronósticos de series de tiempo
        modelo_st = ModeloPrediccion(
            self.df_analisis['deuda_pib_ratio'],
            'Ratio Deuda/PIB (%)'
        )
        modelo_st.dividir_train_test(test_size=8)
        modelo_st.entrenar_todos()
        pron_st = modelo_st.generar_pronostico(periodos=periodos)

        # Luego generar pronósticos ML
        df_ml = self.df_analisis.copy().dropna()
        feature_cols = []
        for col in ['inflacion_yoy_avg', 'TasaMXN', 'TasaUSD', 'tcn_mean']:
            if col in df_ml.columns and df_ml[col].notna().sum() > 10:
                feature_cols.append(col)

        for lag in [1, 2, 4]:
            df_ml[f'deuda_pib_lag{lag}'] = df_ml['deuda_pib_ratio'].shift(lag)
            feature_cols.append(f'deuda_pib_lag{lag}')

        df_ml = df_ml.dropna()

        pron_ml = None
        rmse_gb = 1.0
        if len(df_ml) >= 20:
            X = df_ml[feature_cols]
            y = df_ml['deuda_pib_ratio']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            # Usar Gradient Boosting para híbrido
            modelo_gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
            modelo_gb.fit(X_train, y_train)
            self.modelos_ml['Gradient Boosting'] = modelo_gb
            self.scaler = StandardScaler()
            self.scaler.fit(X_train)

            # Calcular RMSE del modelo
            y_pred = modelo_gb.predict(X_test)
            rmse_gb = np.sqrt(np.mean((y_test - y_pred) ** 2))

            pron_ml = self._pronosticar_ml(df_ml, feature_cols, periodos, 'Gradient Boosting', rmse_gb)

        # Combinar pronósticos (50% ST + 50% ML)
        df_hibrido = pd.DataFrame(index=pron_st.index)
        df_hibrido['Series_Tiempo'] = pron_st['Ensemble']
        if pron_ml is not None:
            df_hibrido['Machine_Learning'] = pron_ml['Ensemble'].values
            df_hibrido['Ensemble'] = 0.5 * df_hibrido['Series_Tiempo'] + 0.5 * df_hibrido['Machine_Learning']
            df_hibrido['Ensemble_std'] = np.sqrt(
                0.25 * pron_st['Ensemble_std']**2 + 0.25 * pron_ml['Ensemble_std'].values**2
            )
        else:
            df_hibrido['Ensemble'] = df_hibrido['Series_Tiempo']
            df_hibrido['Ensemble_std'] = pron_st['Ensemble_std']

        df_hibrido['Ensemble_lower'] = df_hibrido['Ensemble'] - 1.96 * df_hibrido['Ensemble_std']
        df_hibrido['Ensemble_upper'] = df_hibrido['Ensemble'] + 1.96 * df_hibrido['Ensemble_std']

        self.pronosticos_deuda = df_hibrido

        # FSI siempre con series de tiempo
        modelo_fsi = ModeloPrediccion(
            self.df_analisis['FSI'].dropna(),
            'FSI (0-100)'
        )
        modelo_fsi.dividir_train_test(test_size=8)
        modelo_fsi.entrenar_todos()
        self.pronosticos_fsi = modelo_fsi.generar_pronostico(periodos=periodos)

        # Guardar métricas
        metricas_st = pd.DataFrame([
            modelo_st.resultados[m]['metricas']
            for m in modelo_st.resultados
        ])
        metricas_st['serie'] = 'Deuda_PIB_ST'
        metricas_fsi = pd.DataFrame([
            modelo_fsi.resultados[m]['metricas']
            for m in modelo_fsi.resultados
        ])
        metricas_fsi['serie'] = 'FSI'
        self.metricas = pd.concat([metricas_st, metricas_fsi])

        print("  Pronósticos híbridos completados (50% ST + 50% ML)")
        return True

    def obtener_datos_dashboard(self):
        """Retorna todos los datos necesarios para el dashboard"""
        return {
            'df_main': self.df_main,
            'df_fsi': self.df_fsi,
            'df_components': self.df_components,
            'df_analisis': self.df_analisis,
            'pronosticos_deuda': self.pronosticos_deuda,
            'pronosticos_fsi': self.pronosticos_fsi,
            'metricas': self.metricas,
            'proceso': self.proceso,
            'tipo_modelo': self.tipo_modelo
        }

    def cargar_datos_comparativa_banxico(self):
        """Carga datos secundarios de Banxico para comparación con modelo"""
        try:
            datos_comparativa = {}

            # Deuda Externa Total (Banco Mundial)
            path_deuda_bm = self.PATH_SECUNDARIOS / "DeudaExternaTotal_BancoMundial_Mexico.csv"
            if path_deuda_bm.exists():
                df_deuda_bm = pd.read_csv(path_deuda_bm)
                datos_comparativa['deuda_banco_mundial'] = df_deuda_bm
                print(f"  Cargados datos Deuda Externa Banco Mundial: {len(df_deuda_bm)} registros")

            # PIB Corriente (Banco Mundial)
            path_pib_bm = self.PATH_SECUNDARIOS / "PIB_Corriente_BancoMundial_Mexico.csv"
            if path_pib_bm.exists():
                df_pib_bm = pd.read_csv(path_pib_bm)
                datos_comparativa['pib_banco_mundial'] = df_pib_bm
                print(f"  Cargados datos PIB Banco Mundial: {len(df_pib_bm)} registros")

            # Deuda Externa Federal (Banxico) - Trimestral
            path_banxico = self.PATH_SECUNDARIOS / "DeudaExterna_Banxico_Trimestral.csv"
            if path_banxico.exists():
                df_banxico = pd.read_csv(path_banxico, parse_dates=['Fecha'])
                df_banxico.set_index('Fecha', inplace=True)
                datos_comparativa['deuda_banxico'] = df_banxico
                print(f"  Cargados datos Deuda Externa Banxico: {len(df_banxico)} registros")

            return datos_comparativa

        except Exception as e:
            print(f"Error cargando datos comparativa: {e}")
            return {}

    def obtener_comparativa_modelo_vs_real(self):
        """
        Compara pronósticos del modelo con datos reales de Banxico.
        Retorna DataFrame con datos alineados para comparación.
        """
        datos_banxico = self.cargar_datos_comparativa_banxico()

        if not datos_banxico or 'deuda_banxico' not in datos_banxico:
            return None

        df_banxico = datos_banxico['deuda_banxico']

        # Alinear índices temporales
        df_banxico.index = pd.DatetimeIndex(df_banxico.index).to_period('Q').to_timestamp()

        # Crear DataFrame de comparación
        df_comp = pd.DataFrame()

        # Datos del modelo (históricos)
        if self.df_analisis is not None:
            df_modelo = self.df_analisis[['deuda_pib_ratio']].copy()
            df_modelo.columns = ['Modelo_Deuda_PIB']

            # Calcular deuda total del modelo
            deuda_col = self.col_deuda
            if deuda_col in self.df_main.columns:
                df_modelo['Modelo_Deuda_MUSD'] = self.df_main[deuda_col]

            df_comp = df_modelo

        # Datos reales de Banxico
        if 'Deuda_Externa_Total_MUSD' in df_banxico.columns:
            df_real = df_banxico[['Deuda_Externa_Total_MUSD']].copy()
            df_real.columns = ['Real_Deuda_MUSD']
            df_comp = df_comp.join(df_real, how='outer')

        # Calcular métricas de comparación donde hay datos comunes
        if 'Modelo_Deuda_MUSD' in df_comp.columns and 'Real_Deuda_MUSD' in df_comp.columns:
            df_comun = df_comp.dropna(subset=['Modelo_Deuda_MUSD', 'Real_Deuda_MUSD'])
            if len(df_comun) > 0:
                error_abs = np.abs(df_comun['Modelo_Deuda_MUSD'] - df_comun['Real_Deuda_MUSD'])
                error_pct = error_abs / df_comun['Real_Deuda_MUSD'] * 100

                metricas_comp = {
                    'MAE': error_abs.mean(),
                    'MAPE': error_pct.mean(),
                    'Correlacion': df_comun['Modelo_Deuda_MUSD'].corr(df_comun['Real_Deuda_MUSD']),
                    'N_observaciones': len(df_comun)
                }
                return {'datos': df_comp, 'metricas': metricas_comp}

        return {'datos': df_comp, 'metricas': None}

    def obtener_ultimo_periodo(self):
        """Obtiene información del último período"""
        ultimo = self.df_analisis.iloc[-1]

        # Usar columnas detectadas dinámicamente
        pib_col = self.col_pib
        deuda_col = self.col_deuda

        # Obtener período anterior para calcular cambios
        if len(self.df_analisis) > 1:
            anterior = self.df_analisis.iloc[-2]
            deuda_pib_anterior = anterior.get('deuda_pib_ratio', 0)
        else:
            deuda_pib_anterior = 0

        return {
            'fecha': self.df_analisis.index[-1],
            'deuda_pib': ultimo.get('deuda_pib_ratio', 0),
            'deuda_pib_anterior': deuda_pib_anterior,
            'fsi': ultimo.get('FSI', 0),
            'pib': ultimo.get(pib_col, 0) if pib_col in self.df_analisis.columns else 0,
            'deuda': ultimo.get(deuda_col, 0) if deuda_col in self.df_analisis.columns else 0,
            'inflacion': ultimo.get('inflacion_yoy_avg', 0),
            'tasa_mxn': ultimo.get('TasaMXN', 0),
            'tasa_usd': ultimo.get('TasaUSD', 0),
            'tipo_cambio': ultimo.get('tcn_mean', 0),
            'balance_fiscal_pib': ultimo.get('balance_fiscal_pib', 0),
            'crecimiento_pib': ultimo.get('crecimiento_pib', 0),
            'tasa_real': ultimo.get('tasa_real', 0),
            'p_solvencia': ultimo.get('p_solvencia', None),
            'p_externa': ultimo.get('p_externa', None),
            'p_mercado': ultimo.get('p_mercado', None),
            'p_liquidez': ultimo.get('p_liquidez', None),
            'p_fiscal': ultimo.get('p_fiscal', None)
        }

    def obtener_ecuacion_modelo(self):
        """Retorna descripción del modelo de predicción según tipo seleccionado"""
        if self.tipo_modelo == 'series_tiempo':
            return {
                'nombre': 'Series de Tiempo: Ensemble',
                'descripcion': 'Pronóstico ponderado por RMSE con filtro de calidad (umbral 2x). Modelos: ARIMA, SARIMA, ETS.',
                'formula': 'ŷ = w₁·ARIMA + w₂·SARIMA + w₃·ETS',
                'ecuacion_detalle': 'ARIMA(p,d,q): yₜ = c + Σφᵢyₜ₋ᵢ + Σθⱼεₜ₋ⱼ + εₜ\nSARIMA: ARIMA × (P,D,Q)ₛ estacional\nETS: Sₜ = α·yₜ + (1-α)·(Sₜ₋₁ + Tₜ₋₁)',
                'tipo': 'series_tiempo'
            }
        elif self.tipo_modelo == 'machine_learning':
            return {
                'nombre': 'Machine Learning: Mejor Modelo',
                'descripcion': 'Selección automática del mejor modelo ML (Linear, Ridge, Random Forest, Gradient Boosting) por RMSE.',
                'formula': 'ŷ = f(X) donde X = [inflación, tasas, TC, lags]',
                'ecuacion_detalle': 'Linear: ŷ = β₀ + Σβᵢxᵢ\nRidge: min||y - Xβ||² + λ||β||²\nRF: Ensemble de árboles de decisión\nGB: Optimización por gradiente descendente',
                'tipo': 'machine_learning'
            }
        elif self.tipo_modelo == 'hibrido':
            return {
                'nombre': 'Híbrido: Series Tiempo + ML',
                'descripcion': 'Combinación 50/50 de Series de Tiempo (Ensemble) y Machine Learning (Gradient Boosting).',
                'formula': 'ŷ = 0.5·ŷ_ST + 0.5·ŷ_ML',
                'ecuacion_detalle': 'ST: Ensemble(ARIMA, SARIMA, ETS)\nML: Gradient Boosting con features macro\nCombinación: Promedio ponderado 50%-50%',
                'tipo': 'hibrido'
            }
        else:
            return {
                'nombre': 'Modelo No Definido',
                'descripcion': '',
                'formula': '',
                'tipo': self.tipo_modelo
            }

    def clasificar_riesgo_deuda(self, ratio):
        """Clasifica el nivel de riesgo según ratio deuda/PIB"""
        if ratio < 30:
            return 'BAJO', '#2ecc71'
        elif ratio < 50:
            return 'MODERADO', '#f39c12'
        elif ratio < 70:
            return 'ALTO', '#e67e22'
        else:
            return 'CRÍTICO', '#e74c3c'

    def clasificar_riesgo_fsi(self, fsi):
        """Clasifica el estado según FSI"""
        if fsi >= 70:
            return 'SOSTENIBLE', '#2ecc71'
        elif fsi >= 50:
            return 'MODERADO', '#f39c12'
        elif fsi >= 30:
            return 'VULNERABLE', '#e67e22'
        else:
            return 'CRÍTICO', '#e74c3c'

    def calcular_tendencia(self, serie, nombre=''):
        """Calcula la tendencia de una serie"""
        valores = serie.dropna().values
        if len(valores) < 2:
            return 0
        tendencia = np.polyfit(range(len(valores)), valores, 1)[0]
        return tendencia


class ModeloPrediccion:
    """
    Modelo de predicción para series de tiempo.
    Basado en PIB_DEUDA_EXTERNA.ipynb
    Usa ARIMA, SARIMA y ETS con filtro de calidad para ensemble.
    """

    # Umbral para descartar modelos malos (2x el mejor RMSE)
    UMBRAL_DESCARTE = 2.0

    def __init__(self, serie, nombre='Serie'):
        self.serie = serie.dropna().copy()
        if self.serie.index.freq is None:
            self.serie.index = pd.DatetimeIndex(self.serie.index).to_period('Q').to_timestamp()
            self.serie = self.serie.asfreq('QS')
        self.nombre = nombre
        self.modelos = {}
        self.resultados = {}
        self.mejor_modelo = None
        self.test_size = 8

    def dividir_train_test(self, test_size=8):
        """Divide la serie en train y test"""
        self.test_size = test_size
        self.train = self.serie.iloc[:-test_size].copy()
        self.test = self.serie.iloc[-test_size:].copy()
        return self.train, self.test

    def calcular_metricas(self, real, predicho, nombre_modelo):
        """Calcula métricas de error"""
        real_vals = np.array(real.values).flatten()
        pred_vals = np.array(predicho.values if hasattr(predicho, 'values') else predicho).flatten()
        min_len = min(len(real_vals), len(pred_vals))
        real_vals = real_vals[:min_len]
        pred_vals = pred_vals[:min_len]

        mae = mean_absolute_error(real_vals, pred_vals)
        rmse = np.sqrt(mean_squared_error(real_vals, pred_vals))
        mape = mean_absolute_percentage_error(real_vals, pred_vals) * 100

        return {'modelo': nombre_modelo, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

    def entrenar_arima(self):
        """Entrena modelo ARIMA con búsqueda automática de parámetros"""
        try:
            print(f"  Entrenando ARIMA para {self.nombre}...")

            if PMDARIMA_AVAILABLE:
                # Auto ARIMA para encontrar mejores parámetros
                modelo_auto = pm.auto_arima(
                    self.train.values,
                    seasonal=False,  # ARIMA sin estacionalidad
                    stepwise=True,
                    suppress_warnings=True,
                    error_action='ignore',
                    max_p=3, max_q=3, max_d=2
                )
                order = modelo_auto.order
                print(f"    Parámetros óptimos ARIMA: {order}")
            else:
                order = (1, 1, 1)

            modelo = ARIMA(self.train, order=order)
            modelo_fit = modelo.fit()

            pred_test = modelo_fit.forecast(steps=self.test_size)
            pred_test = pd.Series(pred_test.values, index=self.test.index)

            metricas = self.calcular_metricas(self.test, pred_test, 'ARIMA')
            print(f"    ARIMA RMSE: {metricas['RMSE']:.4f}")

            self.modelos['ARIMA'] = modelo_fit
            self.resultados['ARIMA'] = {
                'modelo': modelo_fit,
                'prediccion_test': pred_test,
                'metricas': metricas,
                'orden': order
            }
            return True
        except Exception as e:
            print(f"    Error en ARIMA: {e}")
            return False

    def entrenar_sarima(self):
        """Entrena modelo SARIMA (ARIMA estacional)"""
        try:
            print(f"  Entrenando SARIMA para {self.nombre}...")

            if PMDARIMA_AVAILABLE:
                modelo_auto = pm.auto_arima(
                    self.train.values,
                    seasonal=True,
                    m=4,  # Estacionalidad trimestral
                    stepwise=True,
                    suppress_warnings=True,
                    error_action='ignore',
                    max_p=3, max_q=3, max_P=2, max_Q=2
                )
                order = modelo_auto.order
                seasonal_order = modelo_auto.seasonal_order
                print(f"    Parámetros óptimos SARIMA: {order} x {seasonal_order}")
            else:
                order = (1, 1, 1)
                seasonal_order = (1, 1, 1, 4)

            modelo = SARIMAX(
                self.train, order=order, seasonal_order=seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False
            )
            modelo_fit = modelo.fit(disp=False)

            pred_test = modelo_fit.forecast(steps=self.test_size)
            pred_test = pd.Series(pred_test.values, index=self.test.index)

            metricas = self.calcular_metricas(self.test, pred_test, 'SARIMA')
            print(f"    SARIMA RMSE: {metricas['RMSE']:.4f}")

            self.modelos['SARIMA'] = modelo_fit
            self.resultados['SARIMA'] = {
                'modelo': modelo_fit,
                'prediccion_test': pred_test,
                'metricas': metricas,
                'orden': order,
                'orden_estacional': seasonal_order
            }
            return True
        except Exception as e:
            print(f"    Error en SARIMA: {e}")
            return False

    def entrenar_ets(self):
        """Entrena modelo Exponential Smoothing (ETS)"""
        try:
            print(f"  Entrenando ETS para {self.nombre}...")

            modelo = ExponentialSmoothing(
                self.train, seasonal='additive', seasonal_periods=4,
                trend='additive', damped_trend=True
            )
            modelo_fit = modelo.fit(optimized=True)

            pred_test = modelo_fit.forecast(steps=self.test_size)
            pred_test = pd.Series(pred_test.values, index=self.test.index)

            metricas = self.calcular_metricas(self.test, pred_test, 'ETS')
            print(f"    ETS RMSE: {metricas['RMSE']:.4f}")

            self.modelos['ETS'] = modelo_fit
            self.resultados['ETS'] = {
                'modelo': modelo_fit,
                'prediccion_test': pred_test,
                'metricas': metricas
            }
            return True
        except Exception as e:
            print(f"    Error en ETS: {e}")
            return False

    def entrenar_todos(self):
        """Entrena todos los modelos: ARIMA, SARIMA, ETS"""
        print(f"\nEntrenando modelos para: {self.nombre}")
        print("-" * 50)

        self.entrenar_arima()
        self.entrenar_sarima()
        self.entrenar_ets()

        self.seleccionar_mejor_modelo()
        print("-" * 50)

    def seleccionar_mejor_modelo(self):
        """Selecciona el mejor modelo por RMSE"""
        if not self.resultados:
            return None

        mejor_rmse = float('inf')
        for nombre, resultado in self.resultados.items():
            if resultado['metricas']['RMSE'] < mejor_rmse:
                mejor_rmse = resultado['metricas']['RMSE']
                self.mejor_modelo = nombre

        print(f"  Mejor modelo: {self.mejor_modelo} (RMSE: {mejor_rmse:.4f})")
        return self.mejor_modelo

    def generar_pronostico(self, periodos=12):
        """
        Genera pronóstico futuro con ensemble ponderado.
        Incluye filtro de calidad para descartar modelos con error > 2x el mejor.
        """
        ultima_fecha = self.serie.index[-1]
        fechas_futuras = pd.date_range(
            start=ultima_fecha + pd.DateOffset(months=3),
            periods=periodos, freq='QS'
        )

        pronosticos = {}

        # ARIMA
        if 'ARIMA' in self.resultados:
            try:
                orden = self.resultados['ARIMA'].get('orden', (1, 1, 1))
                modelo = ARIMA(self.serie, order=orden)
                modelo_fit = modelo.fit()
                forecast = modelo_fit.forecast(steps=periodos)
                pronosticos['ARIMA'] = pd.Series(forecast.values, index=fechas_futuras)
            except:
                pass

        # SARIMA
        if 'SARIMA' in self.resultados:
            try:
                orden = self.resultados['SARIMA'].get('orden', (1, 1, 1))
                orden_est = self.resultados['SARIMA'].get('orden_estacional', (1, 1, 1, 4))
                modelo = SARIMAX(
                    self.serie, order=orden, seasonal_order=orden_est,
                    enforce_stationarity=False, enforce_invertibility=False
                )
                modelo_fit = modelo.fit(disp=False)
                forecast = modelo_fit.forecast(steps=periodos)
                pronosticos['SARIMA'] = pd.Series(forecast.values, index=fechas_futuras)
            except:
                pass

        # ETS
        if 'ETS' in self.resultados:
            try:
                modelo = ExponentialSmoothing(
                    self.serie, seasonal='additive', seasonal_periods=4,
                    trend='additive', damped_trend=True
                )
                modelo_fit = modelo.fit(optimized=True)
                forecast = modelo_fit.forecast(steps=periodos)
                pronosticos['ETS'] = pd.Series(forecast.values, index=fechas_futuras)
            except:
                pass

        # Consolidar pronósticos
        df_pronosticos = pd.DataFrame(index=fechas_futuras)
        for nombre, pron in pronosticos.items():
            df_pronosticos[nombre] = pron.values

        # =================================================================
        # FILTRO DE CALIDAD: Descartar modelos con RMSE > 2x el mejor
        # =================================================================
        if self.resultados and pronosticos:
            # Encontrar el mejor RMSE
            mejor_rmse = min(
                self.resultados[m]['metricas']['RMSE']
                for m in pronosticos.keys()
                if m in self.resultados
            )
            umbral_descarte = mejor_rmse * self.UMBRAL_DESCARTE

            # Calcular pesos con filtro de calidad
            pesos = {}
            for nombre in pronosticos.keys():
                if nombre in self.resultados:
                    rmse = self.resultados[nombre]['metricas']['RMSE']
                    if rmse > umbral_descarte:
                        # Modelo muy malo, descartarlo completamente
                        pesos[nombre] = 0
                        print(f"  {nombre} descartado (RMSE {rmse:.4f} > umbral {umbral_descarte:.4f})")
                    else:
                        # Peso inverso al RMSE
                        pesos[nombre] = 1.0 / (rmse + 0.001)

            # Normalizar pesos (solo si hay modelos activos)
            total_peso = sum(pesos.values())
            if total_peso > 0:
                for nombre in pesos:
                    pesos[nombre] /= total_peso

                # Mostrar pesos finales
                pesos_str = ', '.join([f'{k}: {v:.2%}' for k, v in pesos.items() if v > 0])
                print(f"  Pesos del ensemble: {pesos_str}")

                # Calcular ensemble ponderado
                df_pronosticos['Ensemble'] = sum(
                    df_pronosticos[nombre] * pesos[nombre]
                    for nombre in pesos.keys()
                    if pesos[nombre] > 0
                )
            else:
                # Si todos fueron descartados, usar promedio simple
                modelos_disp = list(pronosticos.keys())
                df_pronosticos['Ensemble'] = df_pronosticos[modelos_disp].mean(axis=1)
        else:
            modelos_disp = list(pronosticos.keys())
            df_pronosticos['Ensemble'] = df_pronosticos[modelos_disp].mean(axis=1)

        # Intervalo de confianza basado en la dispersión de modelos activos
        modelos_activos = [m for m in pronosticos.keys() if m in pesos and pesos.get(m, 0) > 0] if 'pesos' in dir() else list(pronosticos.keys())
        if len(modelos_activos) > 1:
            df_pronosticos['Ensemble_std'] = df_pronosticos[modelos_activos].std(axis=1)
        else:
            # Si solo hay un modelo, usar el 5% del valor como std
            df_pronosticos['Ensemble_std'] = df_pronosticos['Ensemble'] * 0.05

        df_pronosticos['Ensemble_lower'] = df_pronosticos['Ensemble'] - 1.96 * df_pronosticos['Ensemble_std']
        df_pronosticos['Ensemble_upper'] = df_pronosticos['Ensemble'] + 1.96 * df_pronosticos['Ensemble_std']

        return df_pronosticos


def ejecutar_procesamiento(proceso='base_2018', tipo_modelo='series_tiempo'):
    """Función principal para ejecutar el procesamiento"""
    procesador = ProcesadorDatos(proceso=proceso, tipo_modelo=tipo_modelo)

    print(f"Cargando datos (proceso: {proceso}, modelo: {tipo_modelo})...")
    if not procesador.cargar_datos():
        raise Exception("Error al cargar datos")

    print("Generando pronósticos...")
    procesador.generar_pronosticos(periodos=12)

    print("Procesamiento completado")
    return procesador


if __name__ == "__main__":
    import sys

    # Argumentos opcionales: proceso y tipo_modelo
    proceso = sys.argv[1] if len(sys.argv) > 1 else 'base_2018'
    tipo_modelo = sys.argv[2] if len(sys.argv) > 2 else 'series_tiempo'

    print(f"\n{'='*60}")
    print(f"PROCESAMIENTO DE DATOS - DASHBOARD DEUDA EXTERNA")
    print(f"Proceso: {proceso} | Modelo: {tipo_modelo}")
    print(f"{'='*60}\n")

    procesador = ejecutar_procesamiento(proceso=proceso, tipo_modelo=tipo_modelo)
    datos = procesador.obtener_datos_dashboard()

    print(f"\n{'='*60}")
    print(f"RESUMEN DE RESULTADOS")
    print(f"{'='*60}")
    print(f"Datos cargados: {len(datos['df_analisis'])} observaciones")
    print(f"Pronósticos deuda: {len(datos['pronosticos_deuda'])} períodos")
    print(f"Pronósticos FSI: {len(datos['pronosticos_fsi'])} períodos")
    print(f"Proceso: {datos['proceso']}")
    print(f"Tipo modelo: {datos['tipo_modelo']}")

    # Mostrar ecuación del modelo
    ecuacion = procesador.obtener_ecuacion_modelo()
    print(f"\nModelo: {ecuacion['nombre']}")
    print(f"Fórmula: {ecuacion['formula']}")
    print(f"{'='*60}\n")
