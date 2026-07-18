"""
Dashboard COVID-19 - Datos sintéticos interactivos
====================================================
Ejecutar con:  streamlit run main_app.py

Genera 10.000 registros sintéticos con 8 columnas de distintos tipos de datos
y ofrece análisis cuantitativo, cualitativo y gráfico con Plotly.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------- #
# Configuración de la página
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Dashboard COVID-19 Sintético",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# 1. Generación / simulación de datos sintéticos
# --------------------------------------------------------------------------- #
@st.cache_data
def generar_datos(n: int = 10_000, semilla: int = 42) -> pd.DataFrame:
    """Crea un DataFrame sintético de COVID con 8 columnas de tipos distintos."""
    rng = np.random.default_rng(semilla)

    regiones = ["Norte", "Sur", "Este", "Oeste", "Centro"]
    generos = ["Femenino", "Masculino", "Otro"]
    severidades = ["Asintomático", "Leve", "Moderado", "Grave", "Crítico"]

    # Fechas de diagnóstico en un rango de ~2 años
    fecha_inicio = pd.Timestamp("2020-03-01")
    dias = rng.integers(0, 730, size=n)
    fechas = fecha_inicio + pd.to_timedelta(dias, unit="D")

    edad = rng.normal(45, 18, n).clip(0, 100).astype(int)          # entero
    # La saturación baja con la edad y la severidad
    sat_o2 = (98 - (edad / 100) * 8 + rng.normal(0, 3, n)).clip(70, 100)  # float

    df = pd.DataFrame(
        {
            "id_paciente": np.arange(1, n + 1),                     # int (identificador)
            "fecha_diagnostico": fechas,                            # datetime
            "region": rng.choice(regiones, n, p=[.22, .18, .20, .20, .20]),  # categórica
            "genero": rng.choice(generos, n, p=[.49, .49, .02]),   # categórica
            "edad": edad,                                          # numérica entera
            "saturacion_o2": sat_o2.round(1),                      # numérica flotante
            "dias_hospital": rng.poisson(4, n),                    # numérica entera (conteo)
            "severidad": rng.choice(severidades, n,                # categórica ordinal
                                    p=[.30, .35, .20, .10, .05]),
        }
    )

    # Variable booleana derivada: fallecido (mayor probabilidad si crítico/grave)
    prob_muerte = df["severidad"].map(
        {"Asintomático": .001, "Leve": .005, "Moderado": .03,
         "Grave": .15, "Crítico": .45}
    )
    df["fallecido"] = rng.random(n) < prob_muerte                  # booleano

    return df


df = generar_datos()

# --------------------------------------------------------------------------- #
# Barra lateral: filtros y controles
# --------------------------------------------------------------------------- #
st.sidebar.title("⚙️ Controles")
st.sidebar.markdown("Personaliza los datos y las gráficas.")

# Filtros de datos
regiones_sel = st.sidebar.multiselect(
    "Región", options=sorted(df["region"].unique()),
    default=sorted(df["region"].unique()),
)
severidad_sel = st.sidebar.multiselect(
    "Severidad", options=df["severidad"].unique().tolist(),
    default=df["severidad"].unique().tolist(),
)
rango_edad = st.sidebar.slider(
    "Rango de edad", int(df["edad"].min()), int(df["edad"].max()),
    (int(df["edad"].min()), int(df["edad"].max())),
)

# Aplicar filtros
mask = (
    df["region"].isin(regiones_sel)
    & df["severidad"].isin(severidad_sel)
    & df["edad"].between(*rango_edad)
)
dff = df[mask]

st.sidebar.markdown("---")
tema_color = st.sidebar.selectbox(
    "Paleta de color",
    ["Plotly", "Viridis", "Turbo", "Blues", "Reds"],
)
paletas = {
    "Plotly": px.colors.qualitative.Plotly,
    "Viridis": px.colors.sequential.Viridis,
    "Turbo": px.colors.sequential.Turbo,
    "Blues": px.colors.sequential.Blues,
    "Reds": px.colors.sequential.Reds,
}

# --------------------------------------------------------------------------- #
# Cabecera
# --------------------------------------------------------------------------- #
st.title("🦠 Dashboard COVID-19 — Datos Sintéticos")
st.caption(
    f"{len(df):,} registros simulados · 8 columnas · "
    f"{len(dff):,} registros tras aplicar filtros"
)

if dff.empty:
    st.warning("No hay registros con los filtros seleccionados.")
    st.stop()

# --------------------------------------------------------------------------- #
# 2. Esquema de métricas
# --------------------------------------------------------------------------- #
tab_resumen, tab_cuant, tab_cual, tab_graf, tab_datos = st.tabs(
    ["📌 Resumen", "📊 Cuantitativo", "🏷️ Cualitativo", "📈 Gráficas", "🗃️ Datos"]
)

# ---- Resumen (KPIs) ------------------------------------------------------- #
with tab_resumen:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Pacientes", f"{len(dff):,}")
    c2.metric("Edad media", f"{dff['edad'].mean():.1f} años")
    c3.metric("Sat. O₂ media", f"{dff['saturacion_o2'].mean():.1f} %")
    tasa_letal = dff["fallecido"].mean() * 100
    c4.metric("Tasa de letalidad", f"{tasa_letal:.2f} %")
    c5.metric("Días hosp. medios", f"{dff['dias_hospital'].mean():.1f}")

    st.markdown("#### Evolución temporal de casos")
    serie = (
        dff.set_index("fecha_diagnostico")
        .resample("W")
        .size()
        .rename("casos")
        .reset_index()
    )
    fig_ts = px.area(serie, x="fecha_diagnostico", y="casos",
                     title="Casos semanales")
    st.plotly_chart(fig_ts, use_container_width=True)

# ---- Estadística cuantitativa -------------------------------------------- #
with tab_cuant:
    st.subheader("Estadística descriptiva (variables numéricas)")
    num_cols = ["edad", "saturacion_o2", "dias_hospital"]
    desc = dff[num_cols].describe().T
    desc["mediana"] = dff[num_cols].median()
    desc["varianza"] = dff[num_cols].var()
    desc["asimetría"] = dff[num_cols].skew()
    desc["curtosis"] = dff[num_cols].kurt()
    st.dataframe(desc.style.format("{:.2f}"), use_container_width=True)

    st.subheader("Matriz de correlación")
    corr = dff[num_cols].corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Correlación de Pearson",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# ---- Estadística cualitativa --------------------------------------------- #
with tab_cual:
    st.subheader("Frecuencias de variables categóricas")
    cat_cols = ["region", "genero", "severidad"]
    col_cat = st.selectbox("Variable categórica", cat_cols)

    tabla = (
        dff[col_cat]
        .value_counts()
        .rename_axis(col_cat)
        .reset_index(name="frecuencia")
    )
    tabla["porcentaje"] = (tabla["frecuencia"] / tabla["frecuencia"].sum() * 100).round(2)

    cA, cB = st.columns([1, 1])
    with cA:
        st.dataframe(tabla, use_container_width=True, hide_index=True)
    with cB:
        fig_pie = px.pie(tabla, names=col_cat, values="frecuencia",
                         hole=.4, title=f"Distribución de {col_cat}")
        st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Tabla de contingencia (cruce de dos variables)")
    c1, c2 = st.columns(2)
    var_x = c1.selectbox("Filas", cat_cols, index=0)
    var_y = c2.selectbox("Columnas", cat_cols, index=2)
    cont = pd.crosstab(dff[var_x], dff[var_y])
    st.dataframe(cont, use_container_width=True)

# ---- Análisis gráfico dinámico ------------------------------------------- #
with tab_graf:
    st.subheader("Gráficas dinámicas personalizables")

    num_cols = ["edad", "saturacion_o2", "dias_hospital"]
    cat_cols = ["region", "genero", "severidad", "fallecido"]

    tipo = st.selectbox(
        "Tipo de gráfica",
        ["Histograma", "Dispersión", "Barras", "Boxplot", "Violín"],
    )

    color_por = st.selectbox("Colorear por (opcional)", ["Ninguno"] + cat_cols)
    color_arg = None if color_por == "Ninguno" else color_por
    seq = paletas[tema_color]
    color_kwargs = (
        {"color_discrete_sequence": seq} if isinstance(seq, list)
        else {"color_continuous_scale": seq}
    )

    fig = None
    if tipo == "Histograma":
        var = st.selectbox("Variable", num_cols)
        nbins = st.slider("Número de bins", 5, 100, 30)
        fig = px.histogram(dff, x=var, color=color_arg, nbins=nbins,
                           marginal="box", title=f"Histograma de {var}",
                           **color_kwargs)
        # Barra de umbral personalizable
        if st.checkbox("Mostrar línea de umbral"):
            umbral = st.slider(
                "Umbral", float(dff[var].min()), float(dff[var].max()),
                float(dff[var].mean()),
            )
            fig.add_vline(x=umbral, line_dash="dash", line_color="red",
                          annotation_text=f"Umbral = {umbral:.1f}")

    elif tipo == "Dispersión":
        cx, cy = st.columns(2)
        var_x = cx.selectbox("Eje X", num_cols, index=0)
        var_y = cy.selectbox("Eje Y", num_cols, index=1)
        tam = st.selectbox("Tamaño por (opcional)", ["Ninguno"] + num_cols)
        size_arg = None if tam == "Ninguno" else tam
        fig = px.scatter(dff, x=var_x, y=var_y, color=color_arg, size=size_arg,
                         opacity=.6, title=f"{var_y} vs {var_x}", **color_kwargs)

    elif tipo == "Barras":
        var_cat = st.selectbox("Variable categórica", cat_cols)
        var_num = st.selectbox("Métrica numérica", num_cols)
        agg = st.selectbox("Agregación", ["mean", "sum", "median", "count"])
        datos = dff.groupby(var_cat)[var_num].agg(agg).reset_index()
        fig = px.bar(datos, x=var_cat, y=var_num, color=var_cat,
                     title=f"{agg} de {var_num} por {var_cat}", **color_kwargs)
        if st.checkbox("Mostrar línea de umbral"):
            umbral = st.slider(
                "Umbral", float(datos[var_num].min()), float(datos[var_num].max()),
                float(datos[var_num].mean()),
            )
            fig.add_hline(y=umbral, line_dash="dash", line_color="red",
                          annotation_text=f"Umbral = {umbral:.1f}")

    elif tipo == "Boxplot":
        var = st.selectbox("Variable numérica", num_cols)
        grupo = st.selectbox("Agrupar por", cat_cols)
        fig = px.box(dff, x=grupo, y=var, color=grupo,
                     title=f"Boxplot de {var} por {grupo}", **color_kwargs)

    elif tipo == "Violín":
        var = st.selectbox("Variable numérica", num_cols)
        grupo = st.selectbox("Agrupar por", cat_cols)
        fig = px.violin(dff, x=grupo, y=var, color=grupo, box=True,
                        title=f"Distribución de {var} por {grupo}", **color_kwargs)

    if fig is not None:
        fig.update_layout(legend_title_text=color_por if color_arg else "")
        st.plotly_chart(fig, use_container_width=True)

# ---- Vista de datos ------------------------------------------------------ #
with tab_datos:
    st.subheader("Datos filtrados")
    st.dataframe(dff, use_container_width=True, height=500)
    st.download_button(
        "⬇️ Descargar CSV",
        dff.to_csv(index=False).encode("utf-8"),
        "covid_sintetico.csv",
        "text/csv",
    )
    with st.expander("Ver tipos de dato de cada columna"):
        tipos = pd.DataFrame(
            {"columna": df.columns, "tipo": df.dtypes.astype(str).values}
        )
        st.dataframe(tipos, use_container_width=True, hide_index=True)