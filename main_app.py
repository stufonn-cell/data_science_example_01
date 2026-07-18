"""
Dashboard Meteorológico - Medellín y Área Metropolitana
========================================================
Análisis sintético por comunas/municipios para apoyar decisiones de la
alcaldía sobre riesgos y desastres (inundaciones, deslizamientos, olas de calor).

EAFIT 2026 · Ciencia de Datos · Profesor Jorge Padilla · Julio de 2026

Ejecutar con:  streamlit run main_app.py
Clave de acceso: 4650
"""
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------------------------------- #
# Configuración de la página
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Meteorología Medellín · EAFIT",
    page_icon="⛅",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLAVE_ACCESO = "4650"


# --------------------------------------------------------------------------- #
# Control de acceso por clave
# --------------------------------------------------------------------------- #
def verificar_clave() -> bool:
    if st.session_state.get("acceso_ok"):
        return True

    st.title("⛅ Dashboard Meteorológico — Medellín y Área Metropolitana")
    st.markdown("#### 🔐 Acceso restringido")
    st.write("Ingrese la clave del dashboard para continuar.")
    clave = st.text_input("Clave", type="password", key="clave_input")
    if st.button("Ingresar"):
        if clave == CLAVE_ACCESO:
            st.session_state["acceso_ok"] = True
            st.rerun()
        else:
            st.error("Clave incorrecta. Intente nuevamente.")
    st.caption("EAFIT 2026 · Ciencia de Datos · Profesor Jorge Padilla · Julio de 2026")
    return False


if not verificar_clave():
    st.stop()


# --------------------------------------------------------------------------- #
# Generación / simulación de datos meteorológicos sintéticos
# --------------------------------------------------------------------------- #
@st.cache_data
def generar_datos(semilla: int = 42) -> pd.DataFrame:
    """
    Serie de tiempo sintética: 10 zonas × 50 días = 500 registros, 10 columnas.
    Variables pensadas para evaluar riesgo de inundación, deslizamiento y calor.
    """
    rng = np.random.default_rng(semilla)

    # Zonas del Valle de Aburrá (municipios + comunas de Medellín)
    zonas = {
        "Medellín - Comuna 1 (Popular)":    {"mun": "Medellín",   "pob": 133000, "alt": 1750},
        "Medellín - Comuna 10 (Candelaria)":{"mun": "Medellín",   "pob": 85000,  "alt": 1490},
        "Medellín - Comuna 13 (San Javier)":{"mun": "Medellín",   "pob": 140000, "alt": 1600},
        "Medellín - Comuna 16 (Belén)":     {"mun": "Medellín",   "pob": 195000, "alt": 1520},
        "Bello":                            {"mun": "Bello",      "pob": 500000, "alt": 1450},
        "Itagüí":                           {"mun": "Itagüí",     "pob": 280000, "alt": 1550},
        "Envigado":                         {"mun": "Envigado",   "pob": 230000, "alt": 1575},
        "Sabaneta":                         {"mun": "Sabaneta",   "pob": 90000,  "alt": 1550},
        "Copacabana":                       {"mun": "Copacabana", "pob": 75000,  "alt": 1454},
        "Girardota":                        {"mun": "Girardota",  "pob": 55000,  "alt": 1425},
    }

    dias = pd.date_range("2026-05-01", periods=50, freq="D")

    filas = []
    for nombre, meta in zonas.items():
        # Tendencia estacional de temperatura (mayo-junio, temporada de lluvias)
        base_temp = 24 - (meta["alt"] - 1450) / 150  # más altura, menos temperatura
        for i, fecha in enumerate(dias):
            estacional = 2 * np.sin(2 * np.pi * i / 15)  # oscilación quincenal
            temp = base_temp + estacional + rng.normal(0, 1.5)
            humedad = np.clip(70 + (base_temp - temp) * 3 + rng.normal(0, 6), 40, 100)
            viento = np.clip(rng.gamma(2.0, 4.0), 0, 45)
            # Lluvia: mayor en temporada, más intensa con humedad alta
            lluvia = max(0, rng.gamma(1.5, 6) * (humedad / 100) - 3)
            nivel_rio = np.clip(1.2 + lluvia * 0.05 + rng.normal(0, 0.15), 0.3, 4.0)

            # Índice de riesgo (0-100) combinando lluvia, nivel de río y viento
            riesgo = np.clip(
                lluvia * 1.8 + (nivel_rio - 1.2) * 25 + viento * 0.4 + rng.normal(0, 4),
                0, 100,
            )
            if riesgo < 25:
                alerta = "Verde"
            elif riesgo < 50:
                alerta = "Amarilla"
            elif riesgo < 75:
                alerta = "Naranja"
            else:
                alerta = "Roja"

            filas.append({
                "fecha": fecha,                                   # datetime
                "municipio": meta["mun"],                         # categórica
                "zona": nombre,                                   # categórica
                "poblacion": meta["pob"],                         # int
                "temperatura_c": round(temp, 1),                  # float
                "humedad_pct": round(humedad, 1),                 # float
                "viento_kmh": round(viento, 1),                   # float
                "precipitacion_mm": round(lluvia, 1),             # float
                "nivel_rio_m": round(nivel_rio, 2),               # float
                "alerta": alerta,                                 # categórica ordinal
            })

    return pd.DataFrame(filas)


df = generar_datos()

# --------------------------------------------------------------------------- #
# Panel izquierdo (branding + controles)
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding:12px 0; line-height:1.5;">
            <span style="font-size:26px; font-weight:800; color:#0033A0;">EAFIT 2026</span><br>
            <span style="font-size:16px; font-weight:600;">Ciencia de Datos</span><br>
            <span style="font-size:14px;">Profesor Jorge Padilla</span><br>
            <span style="font-size:13px; color:gray;">Julio de 2026</span>
        </div>
        <hr>
        """,
        unsafe_allow_html=True,
    )
    st.title("⚙️ Controles")

    municipios_sel = st.multiselect(
        "Municipio", sorted(df["municipio"].unique()),
        default=sorted(df["municipio"].unique()),
    )
    zonas_disp = sorted(df[df["municipio"].isin(municipios_sel)]["zona"].unique())
    zonas_sel = st.multiselect("Zona / Comuna", zonas_disp, default=zonas_disp)

    alertas_sel = st.multiselect(
        "Nivel de alerta", ["Verde", "Amarilla", "Naranja", "Roja"],
        default=["Verde", "Amarilla", "Naranja", "Roja"],
    )

    fmin, fmax = df["fecha"].min().date(), df["fecha"].max().date()
    rango_fecha = st.slider("Rango de fechas", fmin, fmax, (fmin, fmax))

    st.markdown("---")
    tema_color = st.selectbox(
        "Paleta de color", ["Plotly", "Viridis", "Turbo", "Blues", "Reds"]
    )
    if st.button("🚪 Cerrar sesión"):
        st.session_state["acceso_ok"] = False
        st.rerun()

paletas = {
    "Plotly": px.colors.qualitative.Plotly,
    "Viridis": px.colors.sequential.Viridis,
    "Turbo": px.colors.sequential.Turbo,
    "Blues": px.colors.sequential.Blues,
    "Reds": px.colors.sequential.Reds,
}

# Aplicar filtros
mask = (
    df["municipio"].isin(municipios_sel)
    & df["zona"].isin(zonas_sel)
    & df["alerta"].isin(alertas_sel)
    & df["fecha"].dt.date.between(*rango_fecha)
)
dff = df[mask]

# --------------------------------------------------------------------------- #
# Cabecera
# --------------------------------------------------------------------------- #
st.title("⛅ Dashboard Meteorológico — Medellín y Área Metropolitana")
st.caption(
    f"{len(df)} registros sintéticos · 10 columnas · "
    f"{len(dff)} tras filtros · Serie de tiempo mayo–junio 2026"
)

if dff.empty:
    st.warning("No hay registros con los filtros seleccionados.")
    st.stop()

# --------------------------------------------------------------------------- #
# Pestañas de análisis
# --------------------------------------------------------------------------- #
tab_res, tab_cuant, tab_cual, tab_serie, tab_graf, tab_datos = st.tabs(
    ["📌 Resumen", "📊 Cuantitativo", "🏷️ Cualitativo",
     "🕒 Serie de tiempo", "📈 Gráficas", "🗃️ Datos"]
)

# ---- Resumen (KPIs) ------------------------------------------------------- #
with tab_res:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Temp. media", f"{dff['temperatura_c'].mean():.1f} °C")
    c2.metric("Humedad media", f"{dff['humedad_pct'].mean():.0f} %")
    c3.metric("Lluvia total", f"{dff['precipitacion_mm'].sum():.0f} mm")
    c4.metric("Viento máx.", f"{dff['viento_kmh'].max():.0f} km/h")
    alertas_criticas = dff["alerta"].isin(["Naranja", "Roja"]).sum()
    c5.metric("Registros en alerta alta", f"{alertas_criticas}")

    st.markdown("#### 🚨 Zonas con mayor riesgo promedio")
    riesgo_zona = (
        dff.groupby("zona")
        .agg(precip_media=("precipitacion_mm", "mean"),
             nivel_rio_medio=("nivel_rio_m", "mean"),
             poblacion=("poblacion", "first"))
        .sort_values("nivel_rio_medio", ascending=False)
        .reset_index()
    )
    st.dataframe(riesgo_zona.style.format(
        {"precip_media": "{:.1f}", "nivel_rio_medio": "{:.2f}", "poblacion": "{:,.0f}"}
    ), use_container_width=True, hide_index=True)

# ---- Estadística cuantitativa -------------------------------------------- #
with tab_cuant:
    st.subheader("Estadística descriptiva (variables numéricas)")
    num_cols = ["temperatura_c", "humedad_pct", "viento_kmh",
                "precipitacion_mm", "nivel_rio_m", "poblacion"]
    desc = dff[num_cols].describe().T
    desc["mediana"] = dff[num_cols].median()
    desc["varianza"] = dff[num_cols].var()
    desc["asimetría"] = dff[num_cols].skew()
    st.dataframe(desc.style.format("{:.2f}"), use_container_width=True)

    st.subheader("Matriz de correlación")
    corr = dff[num_cols].corr()
    fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto",
                         color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         title="Correlación de Pearson entre variables")
    st.plotly_chart(fig_corr, use_container_width=True)

# ---- Estadística cualitativa --------------------------------------------- #
with tab_cual:
    st.subheader("Frecuencias de variables categóricas")
    cat_cols = ["municipio", "zona", "alerta"]
    col_cat = st.selectbox("Variable categórica", cat_cols)

    tabla = (dff[col_cat].value_counts()
             .rename_axis(col_cat).reset_index(name="frecuencia"))
    tabla["porcentaje"] = (tabla["frecuencia"] / tabla["frecuencia"].sum() * 100).round(2)

    cA, cB = st.columns(2)
    with cA:
        st.dataframe(tabla, use_container_width=True, hide_index=True)
    with cB:
        fig_pie = px.pie(tabla, names=col_cat, values="frecuencia", hole=.4,
                         title=f"Distribución de {col_cat}")
        st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Días por nivel de alerta y municipio")
    cont = pd.crosstab(dff["municipio"], dff["alerta"])
    st.dataframe(cont, use_container_width=True)

# ---- Serie de tiempo ----------------------------------------------------- #
with tab_serie:
    st.subheader("Evolución temporal de una variable")
    num_cols = ["temperatura_c", "humedad_pct", "viento_kmh",
                "precipitacion_mm", "nivel_rio_m"]
    var = st.selectbox("Variable", num_cols)
    agrupar = st.checkbox("Separar por zona", value=True)

    if agrupar:
        serie = dff.groupby(["fecha", "zona"])[var].mean().reset_index()
        fig_ts = px.line(serie, x="fecha", y=var, color="zona", markers=True,
                         title=f"Evolución de {var} por zona")
    else:
        serie = dff.groupby("fecha")[var].mean().reset_index()
        fig_ts = px.line(serie, x="fecha", y=var, markers=True,
                         title=f"Evolución media de {var}")

    # Barra / línea de umbral personalizable
    if st.checkbox("Mostrar umbral de alerta"):
        umbral = st.slider("Umbral", float(dff[var].min()), float(dff[var].max()),
                           float(dff[var].mean()))
        fig_ts.add_hline(y=umbral, line_dash="dash", line_color="red",
                         annotation_text=f"Umbral = {umbral:.1f}")
    st.plotly_chart(fig_ts, use_container_width=True)

# ---- Análisis gráfico dinámico ------------------------------------------- #
with tab_graf:
    st.subheader("Gráficas dinámicas personalizables")
    num_cols = ["temperatura_c", "humedad_pct", "viento_kmh",
                "precipitacion_mm", "nivel_rio_m", "poblacion"]
    cat_cols = ["municipio", "zona", "alerta"]

    tipo = st.selectbox("Tipo de gráfica",
                        ["Histograma", "Dispersión", "Barras", "Boxplot", "Violín"])
    color_por = st.selectbox("Colorear por (opcional)", ["Ninguno"] + cat_cols)
    color_arg = None if color_por == "Ninguno" else color_por
    seq = paletas[tema_color]
    color_kwargs = ({"color_discrete_sequence": seq} if isinstance(seq, list)
                    else {"color_continuous_scale": seq})

    fig = None
    if tipo == "Histograma":
        v = st.selectbox("Variable", num_cols)
        nbins = st.slider("Número de bins", 5, 80, 25)
        fig = px.histogram(dff, x=v, color=color_arg, nbins=nbins, marginal="box",
                           title=f"Histograma de {v}", **color_kwargs)
        if st.checkbox("Mostrar línea de umbral"):
            u = st.slider("Umbral", float(dff[v].min()), float(dff[v].max()),
                          float(dff[v].mean()))
            fig.add_vline(x=u, line_dash="dash", line_color="red",
                          annotation_text=f"Umbral = {u:.1f}")

    elif tipo == "Dispersión":
        cx, cy = st.columns(2)
        vx = cx.selectbox("Eje X", num_cols, index=3)
        vy = cy.selectbox("Eje Y", num_cols, index=4)
        fig = px.scatter(dff, x=vx, y=vy, color=color_arg, opacity=.7,
                         hover_data=["zona", "fecha"],
                         title=f"{vy} vs {vx}", **color_kwargs)

    elif tipo == "Barras":
        vc = st.selectbox("Variable categórica", cat_cols)
        vn = st.selectbox("Métrica numérica", num_cols)
        agg = st.selectbox("Agregación", ["mean", "sum", "median", "max"])
        datos = dff.groupby(vc)[vn].agg(agg).reset_index()
        fig = px.bar(datos, x=vc, y=vn, color=vc,
                     title=f"{agg} de {vn} por {vc}", **color_kwargs)
        if st.checkbox("Mostrar línea de umbral"):
            u = st.slider("Umbral", float(datos[vn].min()), float(datos[vn].max()),
                          float(datos[vn].mean()))
            fig.add_hline(y=u, line_dash="dash", line_color="red",
                          annotation_text=f"Umbral = {u:.1f}")

    elif tipo == "Boxplot":
        v = st.selectbox("Variable numérica", num_cols)
        g = st.selectbox("Agrupar por", cat_cols)
        fig = px.box(dff, x=g, y=v, color=g,
                     title=f"Boxplot de {v} por {g}", **color_kwargs)

    elif tipo == "Violín":
        v = st.selectbox("Variable numérica", num_cols)
        g = st.selectbox("Agrupar por", cat_cols)
        fig = px.violin(dff, x=g, y=v, color=g, box=True,
                        title=f"Distribución de {v} por {g}", **color_kwargs)

    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

# ---- Vista de datos ------------------------------------------------------ #
with tab_datos:
    st.subheader("Datos filtrados")
    st.dataframe(dff, use_container_width=True, height=480)
    st.download_button("⬇️ Descargar CSV",
                       dff.to_csv(index=False).encode("utf-8"),
                       "meteorologia_medellin.csv", "text/csv")
    with st.expander("Ver tipos de dato de cada columna"):
        tipos = pd.DataFrame({"columna": df.columns,
                              "tipo": df.dtypes.astype(str).values})
        st.dataframe(tipos, use_container_width=True, hide_index=True)