# Laboratorio 1 - Series de Tiempo
# Ingreso mensual de viajeros internacionales a Guatemala, 2009 a junio de 2026.

# ============================================================================
# 1. DEPENDENCIAS Y CONFIGURACION
# ============================================================================

import sys
from pathlib import Path
local_deps = Path('.codex_pydeps')
if local_deps.exists() and str(local_deps.resolve()) not in sys.path:
    sys.path.insert(0, str(local_deps.resolve()))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

plt.style.use('seaborn-v0_8-whitegrid')
pd.options.display.float_format = '{:,.2f}'.format


# ============================================================================
# 2. CARGA Y PREPARACION DE DATOS
# ============================================================================

DATA_PATH = Path('Base_Migracion_2009-2026jun.xlsx')
SHEET = 'Datos'

df = pd.read_excel(DATA_PATH, sheet_name=SHEET)
df.columns = [str(c).strip() for c in df.columns]

# Renombrar a ASCII para evitar problemas de codificacion entre sistemas.
column_map = {
    df.columns[0]: 'Anio',
    df.columns[1]: 'MesCod',
    df.columns[2]: 'Mes',
    df.columns[3]: 'Via',
    df.columns[4]: 'Frontera',
    df.columns[5]: 'Pais',
    df.columns[6]: 'Region',
    df.columns[7]: 'RegionDos',
    df.columns[8]: 'RegionesOMT',
    df.columns[9]: 'MCEO',
    df.columns[10]: 'AgrupacionResidencia',
    df.columns[11]: 'TipoViajero',
    df.columns[12]: 'Viajero',
}
df = df.rename(columns=column_map)

# Asegurar tipos utiles para analisis temporal.
df['Fecha'] = pd.to_datetime({'year': df['Anio'], 'month': df['MesCod'], 'day': 1})
df['Viajero'] = pd.to_numeric(df['Viajero'], errors='coerce')

print("Forma del dataframe:", df.shape)
print("\nPrimeras filas:")
print(df.head())
print("\nTipos de datos:")
print(df.dtypes.to_frame('tipo'))

# ============================================================================
# 3. ANALISIS DE CALIDAD DE DATOS
# ============================================================================

resumen_calidad = pd.DataFrame({
    'faltantes': df.isna().sum(),
    'faltantes_pct': (df.isna().mean() * 100).round(2),
    'unicos': df.nunique(dropna=False),
})
print(f"\nFilas duplicadas exactas: {df.duplicated().sum():,}")
print("\nResumen de calidad:")
print(resumen_calidad)

serie_calendario = df.groupby('Fecha')['Viajero'].sum().asfreq('MS')
print('\nRango temporal:', df['Fecha'].min().date(), 'a', df['Fecha'].max().date())
print('Meses calendario en la serie total:', serie_calendario.shape[0])
print('Meses faltantes de calendario:', int(serie_calendario.isna().sum()))


# ============================================================================
# 4. ANALISIS EXPLORATORIO DEL CONJUNTO DE DATOS
# ============================================================================

def top_table(col, n=10):
    out = (df.groupby(col, dropna=False)['Viajero']
             .sum()
             .sort_values(ascending=False)
             .head(n)
             .reset_index())
    out['participacion_pct'] = out['Viajero'] / df['Viajero'].sum() * 100
    return out

eda_tables = {
    'Top paises': top_table('Pais', 10),
    'Top regiones dos': top_table('RegionDos', 10),
    'Vias de ingreso': top_table('Via', 10),
    'Top fronteras': top_table('Frontera', 10),
    'Tipo de viajero': top_table('TipoViajero', 10),
}

for name, table in eda_tables.items():
    print('\n' + name)
    print(table.to_string(index=False))

# Visualizacion del total mensual
monthly_total = df.groupby('Fecha')['Viajero'].sum().asfreq('MS')
annual_total = df.groupby('Anio')['Viajero'].sum()

fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
monthly_total.plot(ax=axes[0], color='#1f77b4', linewidth=1.8)
axes[0].set_title('Total mensual de viajeros internacionales')
axes[0].set_xlabel('')
axes[0].set_ylabel('Viajeros')

annual_total.plot(kind='bar', ax=axes[1], color='#4c78a8')
axes[1].set_title('Total anual de viajeros')
axes[1].set_xlabel('Anio')
axes[1].set_ylabel('Viajeros')
plt.savefig('01_total_temporal.png', dpi=100, bbox_inches='tight')
plt.show()

print('\nMes minimo:', monthly_total.idxmin().date(), f"{monthly_total.min():,.0f}")
print('Mes maximo:', monthly_total.idxmax().date(), f"{monthly_total.max():,.0f}")

# Visualizacion por categorias
fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

for ax, (title, col, n) in zip(
    axes.ravel(),
    [('Paises con mayor cantidad de viajeros', 'Pais', 10),
     ('Regiones con mayor cantidad de viajeros', 'RegionDos', 10),
     ('Vias de ingreso', 'Via', 5),
     ('Fronteras mas utilizadas', 'Frontera', 10)]
):
    t = top_table(col, n).sort_values('Viajero')
    ax.barh(t[col].astype(str), t['Viajero'], color='#3a7ca5')
    ax.set_title(title)
    ax.set_xlabel('Viajeros acumulados')
plt.savefig('02_categorias_barh.png', dpi=100, bbox_inches='tight')
plt.show()

# Deteccion de valores atipicos
q1, q3 = monthly_total.quantile([0.25, 0.75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outliers_mensuales = monthly_total[(monthly_total < lower) | (monthly_total > upper)].to_frame('viajeros')

print(f'\nLimite inferior IQR: {lower:,.0f}')
print(f'Limite superior IQR: {upper:,.0f}')
print("\nValores atipicos (IQR):")
print(outliers_mensuales)

fig, ax = plt.subplots(figsize=(12, 4))
ax.boxplot(monthly_total.dropna(), vert=False)
ax.set_title('Distribucion del total mensual y deteccion de atipicos')
ax.set_xlabel('Viajeros')
plt.savefig('03_boxplot_outliers.png', dpi=100, bbox_inches='tight')
plt.show()


# ============================================================================
# 5. CONSTRUCCION DE SERIES MENSUALES
# ============================================================================

series = {
    'Total': monthly_total,
}

# Categoria seleccionada: vias de ingreso.
via_series = (df.pivot_table(index='Fecha', columns='Via', values='Viajero', aggfunc='sum')
                .asfreq('MS')
                .fillna(0))
for col in via_series.columns:
    series[col] = via_series[col]

# Tambien se construyen otras categorias pedidas para dejar el avance encaminado.
top_paises = top_table('Pais', 3)['Pais'].tolist()
pais_series = (df[df['Pais'].isin(top_paises)]
               .pivot_table(index='Fecha', columns='Pais', values='Viajero', aggfunc='sum')
               .asfreq('MS')
               .fillna(0))

top_regiones = top_table('RegionDos', 3)['RegionDos'].tolist()
region_series = (df[df['RegionDos'].isin(top_regiones)]
                 .pivot_table(index='Fecha', columns='RegionDos', values='Viajero', aggfunc='sum')
                 .asfreq('MS')
                 .fillna(0))

print('Top 3 paises:', top_paises)
print('Top 3 regiones:', top_regiones)
print('Series de vias:', list(via_series.columns))

def train_test_split_ts(s, train_ratio=0.70):
    n_train = int(len(s) * train_ratio)
    return s.iloc[:n_train], s.iloc[n_train:]

split_info = []
for name, s in series.items():
    train, test = train_test_split_ts(s)
    split_info.append({
        'serie': name,
        'inicio': s.index.min().date(),
        'fin': s.index.max().date(),
        'frecuencia': pd.infer_freq(s.index),
        'n_meses': len(s),
        'train_inicio': train.index.min().date(),
        'train_fin': train.index.max().date(),
        'test_inicio': test.index.min().date(),
        'test_fin': test.index.max().date(),
    })
print("\nDivision entrenamiento/prueba:")
print(pd.DataFrame(split_info).to_string(index=False))


# ============================================================================
# 6. FUNCIONES PARA ANALIZAR SERIES
# ============================================================================

def mae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def seasonal_naive_forecast(train, steps, season_length=12):
    reps = int(np.ceil(steps / season_length))
    vals = np.tile(train.iloc[-season_length:].values, reps)[:steps]
    return pd.Series(vals, index=pd.date_range(train.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq='MS'))

def choose_d(train, max_d=2):
    current = train.copy()
    for d in range(max_d + 1):
        clean = current.dropna()
        if len(clean) > 20:
            pvalue = adfuller(clean, autolag='AIC')[1]
            if pvalue < 0.05:
                return d, pvalue
        current = current.diff()
    return max_d, adfuller(train.diff(max_d).dropna(), autolag='AIC')[1]

def has_year(s, year):
    return (s.index.year == year).any()

def analyze_series(name, s, seasonal_period=12, show_plots=True):
    s = s.asfreq('MS').astype(float)
    train, test = train_test_split_ts(s)
    d, adf_p_after = choose_d(train)
    adf_original = adfuller(train.dropna(), autolag='AIC')
    transformed = np.log1p(s) if (s >= 0).all() else s.copy()
    stl = STL(transformed, period=seasonal_period, robust=True).fit()

    if show_plots:
        fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True, constrained_layout=True)
        s.plot(ax=axes[0], color='#1f77b4')
        axes[0].axvline(test.index[0], color='black', linestyle='--', linewidth=1, label='Inicio prueba')
        axes[0].set_title(f'{name}: serie mensual')
        axes[0].legend()
        stl.trend.plot(ax=axes[1], color='#3a7ca5')
        axes[1].set_title('Tendencia STL')
        stl.seasonal.plot(ax=axes[2], color='#59a14f')
        axes[2].set_title('Componente estacional STL')
        stl.resid.plot(ax=axes[3], color='#e15759')
        axes[3].set_title('Residuo STL')
        plt.savefig(f'04_{name}_descomposicion.png', dpi=100, bbox_inches='tight')
        plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
        plot_acf(train.dropna(), lags=36, ax=axes[0])
        axes[0].set_title(f'{name}: ACF')
        plot_pacf(train.dropna(), lags=36, ax=axes[1], method='ywm')
        axes[1].set_title(f'{name}: PACF')
        plt.savefig(f'05_{name}_acf_pacf.png', dpi=100, bbox_inches='tight')
        plt.show()

    candidate_orders = [(0, d, 1), (1, d, 0), (1, d, 1), (2, d, 1), (1, d, 2)]
    rows = []
    preds = {}

    for order in candidate_orders:
        try:
            model = SARIMAX(train, order=order, enforce_stationarity=False, enforce_invertibility=False)
            fit = model.fit(disp=False)
            pred = fit.forecast(len(test))
            rows.append({
                'modelo': f'ARIMA{order}',
                'p': order[0], 'd': order[1], 'q': order[2],
                'AIC': fit.aic,
                'BIC': fit.bic,
                'MAE': mae(test, pred),
                'RMSE': rmse(test, pred),
            })
            preds[f'ARIMA{order}'] = pred
        except Exception:
            rows.append({'modelo': f'ARIMA{order}', 'p': order[0], 'd': order[1], 'q': order[2], 'AIC': np.nan, 'BIC': np.nan, 'MAE': np.nan, 'RMSE': np.nan})

    try:
        hw_fit = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12, initialization_method='estimated').fit(optimized=True)
        hw_pred = hw_fit.forecast(len(test))
        rows.append({'modelo': 'Holt-Winters aditivo', 'p': np.nan, 'd': np.nan, 'q': np.nan, 'AIC': getattr(hw_fit, 'aic', np.nan), 'BIC': getattr(hw_fit, 'bic', np.nan), 'MAE': mae(test, hw_pred), 'RMSE': rmse(test, hw_pred)})
        preds['Holt-Winters aditivo'] = hw_pred
    except Exception:
        pass

    try:
        ses_fit = SimpleExpSmoothing(train, initialization_method='estimated').fit(optimized=True)
        ses_pred = ses_fit.forecast(len(test))
        rows.append({'modelo': 'Suavizamiento exponencial simple', 'p': np.nan, 'd': np.nan, 'q': np.nan, 'AIC': getattr(ses_fit, 'aic', np.nan), 'BIC': getattr(ses_fit, 'bic', np.nan), 'MAE': mae(test, ses_pred), 'RMSE': rmse(test, ses_pred)})
        preds['Suavizamiento exponencial simple'] = ses_pred
    except Exception:
        pass

    sn_pred = seasonal_naive_forecast(train, len(test), season_length=12)
    rows.append({'modelo': 'Seasonal naive', 'p': np.nan, 'd': np.nan, 'q': np.nan, 'AIC': np.nan, 'BIC': np.nan, 'MAE': mae(test, sn_pred), 'RMSE': rmse(test, sn_pred)})
    preds['Seasonal naive'] = sn_pred

    metrics = pd.DataFrame(rows).sort_values(['RMSE', 'MAE'], na_position='last').reset_index(drop=True)
    best_name = metrics.iloc[0]['modelo']

    if show_plots:
        fig, ax = plt.subplots(figsize=(13, 5))
        train.plot(ax=ax, label='Entrenamiento', color='#4c78a8')
        test.plot(ax=ax, label='Prueba', color='#111111')
        preds[best_name].plot(ax=ax, label=f'Prediccion: {best_name}', color='#f28e2b')
        ax.set_title(f'{name}: mejor prediccion en prueba')
        ax.set_ylabel('Viajeros')
        ax.legend()
        plt.savefig(f'06_{name}_mejor_prediccion.png', dpi=100, bbox_inches='tight')
        plt.show()

    stl_raw = STL(s, period=seasonal_period, robust=True).fit()
    resid_var = np.nanvar(stl_raw.resid)
    season_strength = max(0, 1 - resid_var / np.nanvar(stl_raw.resid + stl_raw.seasonal))
    trend = stl_raw.trend.dropna()
    slope = np.polyfit(np.arange(len(trend)), trend.values, 1)[0]
    volatility = s.pct_change().replace([np.inf, -np.inf], np.nan).std()
    avg_2019 = s[s.index.year == 2019].mean() if has_year(s, 2019) else np.nan
    avg_2020 = s[s.index.year == 2020].mean() if has_year(s, 2020) else np.nan
    pandemic_drop = (avg_2020 / avg_2019 - 1) * 100 if avg_2019 and not np.isnan(avg_2019) else np.nan

    summary = {
        'serie': name,
        'inicio': s.index.min().date(),
        'fin': s.index.max().date(),
        'frecuencia': pd.infer_freq(s.index),
        'ADF_p_original_train': adf_original[1],
        'd_recomendado': d,
        'ADF_p_tras_diferenciar': adf_p_after,
        'fuerza_estacional': season_strength,
        'pendiente_tendencia_mensual': slope,
        'volatilidad_retornos_mensuales': volatility,
        'caida_promedio_2020_vs_2019_pct': pandemic_drop,
        'mejor_modelo_RMSE': best_name,
    }
    return summary, metrics, preds


# ============================================================================
# 7. ANALISIS DE LA SERIE OBLIGATORIA: TOTAL MENSUAL
# ============================================================================

print("\n" + "="*80)
print("ANALISIS DE LA SERIE TOTAL MENSUAL")
print("="*80)
summary_total, metrics_total, preds_total = analyze_series('Total', series['Total'])
print("\nResumen de la serie Total:")
print(pd.DataFrame([summary_total]).to_string(index=False))
print("\nMetricas de modelos para Total:")
print(metrics_total.to_string(index=False))


# ============================================================================
# 8. SERIE SELECCIONADA 1: VIA AEREA
# ============================================================================

print("\n" + "="*80)
print("ANALISIS DE LA SERIE AEREA")
print("="*80)
aerea_key = next(k for k in series.keys() if k.lower().startswith('a'))
summary_aerea, metrics_aerea, preds_aerea = analyze_series('Aerea', series[aerea_key])
print("\nResumen de la serie Aerea:")
print(pd.DataFrame([summary_aerea]).to_string(index=False))
print("\nMetricas de modelos para Aerea:")
print(metrics_aerea.to_string(index=False))


# ============================================================================
# 9. SERIE SELECCIONADA 2: VIA TERRESTRE
# ============================================================================

print("\n" + "="*80)
print("ANALISIS DE LA SERIE TERRESTRE")
print("="*80)
summary_terrestre, metrics_terrestre, preds_terrestre = analyze_series('Terrestre', series['Terrestre'])
print("\nResumen de la serie Terrestre:")
print(pd.DataFrame([summary_terrestre]).to_string(index=False))
print("\nMetricas de modelos para Terrestre:")
print(metrics_terrestre.to_string(index=False))


# ============================================================================
# 10. COMPARACION ESTADISTICA DE LAS SERIES ANALIZADAS
# ============================================================================

print("\n" + "="*80)
print("COMPARACION ESTADISTICA DE LAS TRES SERIES")
print("="*80)

comparacion = pd.DataFrame([summary_total, summary_aerea, summary_terrestre])
print("\nComparacion de caracteristicas:")
print(comparacion[[
    'serie', 'fuerza_estacional', 'pendiente_tendencia_mensual',
    'volatilidad_retornos_mensuales', 'caida_promedio_2020_vs_2019_pct',
    'mejor_modelo_RMSE'
]].sort_values('serie').to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
comparacion.set_index('serie')['fuerza_estacional'].plot(kind='bar', ax=axes[0], color='#59a14f')
axes[0].set_title('Fuerza estacional')
comparacion.set_index('serie')['pendiente_tendencia_mensual'].plot(kind='bar', ax=axes[1], color='#4c78a8')
axes[1].set_title('Pendiente de tendencia')
comparacion.set_index('serie')['volatilidad_retornos_mensuales'].plot(kind='bar', ax=axes[2], color='#e15759')
axes[2].set_title('Volatilidad mensual')
plt.savefig('07_comparacion_series.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("ANALISIS COMPLETADO - Revisar imagenes PNG generadas")
print("="*80)
