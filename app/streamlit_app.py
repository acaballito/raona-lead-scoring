"""Raona Lead Scoring - B2B Prospecting Intelligence

Streamlit app con estetica Anthropic (ivory, serif headings, minimalista).
Tres pestanas: Dashboard, Lead Scorer y Batch Scorer.

Uso:
    streamlit run streamlit_app.py
"""
import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --- Page config ---
st.set_page_config(
    page_title="Raona Lead Scoring",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={},
)

# --- Anthropic-inspired CSS ---
st.markdown("""
<style>
    /* Hide deploy button, hamburger, footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    [data-testid="stToolbar"] {display: none;}

    /* Global background and typography */
    .stApp {
        background-color: #FAFAF7;
    }
    html, body, [class*="css"] {
        font-family: 'Tiempos Text', 'Georgia', 'Times New Roman', serif;
        color: #141413;
    }

    /* Headings: serif, tight */
    h1 {
        font-family: 'Copernicus', 'Georgia', serif !important;
        font-weight: 700 !important;
        color: #141413 !important;
        letter-spacing: -0.02em !important;
        font-size: 2.4rem !important;
        line-height: 1.1 !important;
        margin-bottom: 0.3rem !important;
    }
    h2 {
        font-family: 'Copernicus', 'Georgia', serif !important;
        font-weight: 600 !important;
        color: #141413 !important;
        font-size: 1.5rem !important;
        letter-spacing: -0.01em !important;
        margin-top: 2rem !important;
    }
    h3 {
        font-family: 'Copernicus', 'Georgia', serif !important;
        font-weight: 600 !important;
        color: #141413 !important;
        font-size: 1.15rem !important;
    }

    /* Body text */
    p, .stMarkdown, label {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', sans-serif !important;
        color: #3D3D3A !important;
        font-size: 0.95rem !important;
        line-height: 1.6 !important;
    }

    /* Subtitle / lead text */
    .lead-text {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        color: #6B6B66;
        font-size: 1.1rem;
        line-height: 1.6;
        margin-bottom: 2.5rem;
        max-width: 700px;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E8E8E3;
        border-radius: 8px;
        padding: 1.2rem 1rem;
    }
    [data-testid="stMetricLabel"] {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        color: #8B8B85 !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Copernicus', 'Georgia', serif !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #141413 !important;
    }
    [data-testid="stMetricDelta"] {
        font-family: -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 0.8rem !important;
    }

    /* Dividers */
    hr {
        border: none;
        border-top: 1px solid #E8E8E3;
        margin: 2rem 0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #F0F0EB;
        border-right: 1px solid #E8E8E3;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
        font-size: 1.1rem !important;
    }

    /* Form styling */
    .stForm {
        background: #FFFFFF;
        border: 1px solid #E8E8E3;
        border-radius: 8px;
        padding: 1.5rem;
    }

    /* Button -- Nuclio gold accent */
    .stFormSubmitButton button {
        background-color: #FFC630 !important;
        color: #141413 !important;
        border: none !important;
        border-radius: 6px !important;
        font-family: -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        padding: 0.6rem 2rem !important;
        transition: background-color 0.2s !important;
    }
    .stFormSubmitButton button:hover {
        background-color: #EDB82D !important;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border: 1px solid #E8E8E3;
        border-radius: 8px;
    }

    /* Radio buttons in sidebar */
    .stRadio label {
        font-family: -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Override Streamlit red/green delta colors */
    [data-testid="stMetricDelta"] svg {
        display: none;
    }
    [data-testid="stMetricDelta"] > div {
        color: #3D3D3A !important;
    }

    /* Remove red from any Streamlit element */
    .stAlert, [data-baseweb="notification"] {
        background-color: #FFF9E6 !important;
        border-color: #FFC630 !important;
        color: #141413 !important;
    }

    /* Top border accent */
    .top-accent {
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, #FFC630 0%, #141413 50%, #E8E8E3 100%);
        margin-bottom: 2rem;
    }

    /* Section label */
    .section-label {
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #8B8B85;
        margin-bottom: 0.5rem;
    }

    /* Recommendation cards */
    .rec-card {
        background: #FFFFFF;
        border: 1px solid #E8E8E3;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 0.5rem 0;
    }
    .rec-card-high {
        border-left: 4px solid #FFC630;
    }
    .rec-card-mid {
        border-left: 4px solid #141413;
    }
    .rec-card-low {
        border-left: 4px solid #D4D4CF;
    }

    /* Slider -- gold accent (multiple selectors for compatibility) */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #FFC630 !important;
        border-color: #FFC630 !important;
    }
    .stSlider [data-baseweb="slider"] div[data-testid="stTickBar"] {
        background: #FFC630 !important;
    }
    div[data-baseweb="slider"] > div > div > div[role="progressbar"] > div {
        background-color: #FFC630 !important;
    }
    /* Slider thumb and track */
    .stSlider div[role="slider"] {
        background: #FFC630 !important;
    }
    .stSlider [data-testid="stThumbValue"] {
        color: #141413 !important;
    }
    /* Slider filled track */
    .stSlider div[data-baseweb="slider"] div div div {
        background-color: #FFC630 !important;
    }
    /* Force override on ALL slider inner colored divs */
    [data-testid="stSlider"] div[role="slider"] {
        background: #FFC630 !important;
        border-color: #EDB82D !important;
    }
    [data-testid="stSlider"] div[data-testid="stTickBar"] > div {
        background: #FFC630 !important;
    }

    /* Selectbox / input focus */
    [data-baseweb="select"] [data-baseweb="input"]:focus-within,
    .stNumberInput input:focus {
        border-color: #FFC630 !important;
        box-shadow: 0 0 0 1px #FFC630 !important;
    }

    /* Checkbox gold (multiple selectors) */
    .stCheckbox [data-testid="stCheckbox"] input:checked + div {
        background-color: #FFC630 !important;
        border-color: #FFC630 !important;
    }
    .stCheckbox label span[data-testid="stCheckbox"] {
        color: #FFC630 !important;
    }
    /* Force checkbox checked state */
    input[type="checkbox"]:checked + label > span:first-child,
    [data-testid="stCheckbox"] > label > div:first-child {
        --checkbox-checked-bg: #FFC630 !important;
    }
    .st-emotion-cache-1inwz65,
    .st-emotion-cache-16txtl3 {
        color: #FFC630 !important;
    }
    /* Broadest checkbox override */
    [data-testid="stCheckbox"] svg {
        fill: #FFC630 !important;
        color: #FFC630 !important;
    }
    [data-testid="stCheckbox"] [aria-checked="true"] {
        background-color: #FFC630 !important;
        border-color: #FFC630 !important;
    }
    /* Override Streamlit's primary color everywhere */
    :root {
        --primary-color: #FFC630 !important;
    }

    /* Override any red elements from Streamlit */
    .element-container .stException,
    .stError {
        background-color: #FFF9E6 !important;
        border-color: #FFC630 !important;
    }

    /* Hide fullscreen buttons on charts */
    button[title="View fullscreen"] {display: none;}

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid #E8E8E3;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.03em !important;
        color: #8B8B85 !important;
        padding: 0.8rem 1.5rem !important;
        border: none !important;
        background: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        color: #141413 !important;
        border-bottom: 3px solid #FFC630 !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #FFC630 !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# --- Plotly theme matching Anthropic ---
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="-apple-system, BlinkMacSystemFont, sans-serif", color="#3D3D3A", size=12),
    margin=dict(t=20, b=40, l=50, r=20),
    colorway=["#141413", "#FFC630", "#6B6B66", "#B5B5AE", "#D4D4CF"],
    xaxis=dict(gridcolor="#E8E8E3", linecolor="#E8E8E3"),
    yaxis=dict(gridcolor="#E8E8E3", linecolor="#E8E8E3"),
)

ACCENT_DARK = "#141413"
ACCENT_MID = "#6B6B66"
ACCENT_LIGHT = "#B5B5AE"
ACCENT_GOLD = "#FFC630"
ACCENT_GOLD_DARK = "#EDB82D"

# --- Load data and models ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "models")
DATA_DIR = os.path.join(APP_DIR, "data")


@st.cache_data
def load_data():
    df = pd.read_parquet(os.path.join(DATA_DIR, "sample_contacts.parquet"))
    daily = pd.read_parquet(os.path.join(DATA_DIR, "daily_analytics_ES.parquet"))
    return df, daily


@st.cache_resource
def load_models():
    with open(os.path.join(MODEL_DIR, "preprocessor.pkl"), "rb") as f:
        preprocessor = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "lead_scorer.pkl"), "rb") as f:
        lead_scorer = pickle.load(f)
    # Modelo robusto (produccion, sin feature con leakage)
    robust_path = os.path.join(MODEL_DIR, "lead_scorer_robust.pkl")
    lead_scorer_robust = None
    if os.path.exists(robust_path):
        with open(robust_path, "rb") as f:
            lead_scorer_robust = pickle.load(f)
    # Preprocessor robusto
    robust_prep_path = os.path.join(MODEL_DIR, "preprocessor_robust.pkl")
    preprocessor_robust = None
    if os.path.exists(robust_prep_path):
        with open(robust_prep_path, "rb") as f:
            preprocessor_robust = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "clustering.pkl"), "rb") as f:
        clustering = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "feature_names.pkl"), "rb") as f:
        feature_names = pickle.load(f)
    return preprocessor, lead_scorer, lead_scorer_robust, preprocessor_robust, clustering, feature_names


df, daily = load_data()
preprocessor, lead_scorer, lead_scorer_robust, preprocessor_robust, clustering_bundle, FEATURE_COLS_RAW = load_models()

# Resolve feature names (may be dict with 'complete' and 'robust' keys)
if isinstance(FEATURE_COLS_RAW, dict):
    FEATURE_COLS = FEATURE_COLS_RAW.get("robust", FEATURE_COLS_RAW.get("complete", []))
    FEATURE_COLS_COMPLETE = FEATURE_COLS_RAW.get("complete", FEATURE_COLS)
    FEATURE_COLS_ROBUST = FEATURE_COLS_RAW.get("robust", FEATURE_COLS)
else:
    FEATURE_COLS = FEATURE_COLS_RAW
    FEATURE_COLS_COMPLETE = FEATURE_COLS_RAW
    FEATURE_COLS_ROBUST = FEATURE_COLS_RAW

# Global stats (used across pages)
TOTAL_CONTACTS = len(df)
TOTAL_REPLIED = int(df["target_replied"].sum())
GLOBAL_REPLY_RATE = TOTAL_REPLIED / TOTAL_CONTACTS * 100

# Cluster profile labels
CLUSTER_NAMES = {
    0: "SMB Microsoft FIT+",
    1: "Enterprise Microsoft FIT+",
    2: "Senior Non-FIT Responders",
    3: "Non-Microsoft FIT+",
}

# --- Navigation via tabs ---
tab_dashboard, tab_scorer, tab_batch = st.tabs(["Dashboard", "Lead Scorer", "Batch Scorer"])

# Cluster label helper
def cluster_label(c):
    return CLUSTER_NAMES.get(c, f"Segmento {c}")


# =====================================================
# PAGE 1: DASHBOARD
# =====================================================
with tab_dashboard:
    st.markdown('<div class="top-accent"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Vista general</div>', unsafe_allow_html=True)
    st.title("B2B Prospecting Intelligence")
    st.markdown(
        '<div class="lead-text">'
        "5,987 contactos analizados revelan patrones sobre quien responde, "
        "por que canal y cuando. El modelo de lead scoring identifica los mejores leads "
        "con un lift de 2.6x sobre la seleccion aleatoria."
        "</div>",
        unsafe_allow_html=True,
    )

    # --- KPIs ---
    ln_replied = int(df["target_replied_linkedin"].sum())
    em_replied = int(df["target_replied_email"].sum())

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Contactos", f"{TOTAL_CONTACTS:,}")
    col2.metric("Respuestas", f"{TOTAL_REPLIED:,}")
    col3.metric("Tasa respuesta", f"{GLOBAL_REPLY_RATE:.1f}%")
    col4.metric("LinkedIn", f"{ln_replied:,}")
    col5.metric("Email", f"{em_replied:,}")

    st.markdown("---")

    # --- Row 1: Score distribution + Lift ---
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown('<div class="section-label">Distribucion</div>', unsafe_allow_html=True)
        st.markdown("### Distribucion de lead scores")
        fig = go.Figure()
        neg_scores = df[df["target_replied"] == 0]["lead_score"]
        pos_scores = df[df["target_replied"] == 1]["lead_score"]
        fig.add_trace(go.Histogram(
            x=neg_scores, nbinsx=50, name="No respondio",
            marker_color=ACCENT_LIGHT, opacity=0.7,
        ))
        fig.add_trace(go.Histogram(
            x=pos_scores, nbinsx=50, name="Respondio",
            marker_color=ACCENT_GOLD, opacity=0.9,
        ))
        fig.update_layout(**PLOTLY_LAYOUT, barmode="overlay", height=340,
                          legend=dict(orientation="h", yanchor="top", y=1.12, x=0))
        fig.update_xaxes(title_text="Lead score")
        fig.update_yaxes(title_text="Cantidad")
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        st.markdown('<div class="section-label">Rendimiento del modelo</div>', unsafe_allow_html=True)
        st.markdown("### Curva de ganancia acumulada")
        sorted_idx = np.argsort(-df["lead_score"].values)
        y_sorted = df["target_replied"].values[sorted_idx]
        n = len(y_sorted)
        cum_pos = np.cumsum(y_sorted)
        total_pos = y_sorted.sum()
        pct_x = np.arange(1, n + 1) / n * 100
        pct_y = cum_pos / total_pos * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pct_x, y=pct_y, mode="lines", name="Modelo",
            line=dict(color=ACCENT_GOLD, width=2.5),
        ))
        fig.add_trace(go.Scatter(
            x=[0, 100], y=[0, 100], mode="lines", name="Aleatorio",
            line=dict(color=ACCENT_LIGHT, width=1.5, dash="dash"),
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=340,
                          legend=dict(orientation="h", yanchor="top", y=1.12, x=0))
        fig.update_xaxes(title_text="% de contactos (ordenados por score)")
        fig.update_yaxes(title_text="% de respuestas capturadas")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- Row 2: Channel + Timing ---
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown('<div class="section-label">Analisis de canal</div>', unsafe_allow_html=True)
        st.markdown("### Tasa de respuesta por canal")
        channel_data = pd.DataFrame({
            "Channel": ["LinkedIn", "Email"],
            "Replies": [ln_replied, em_replied],
            "Rate": [ln_replied / TOTAL_CONTACTS * 100, em_replied / TOTAL_CONTACTS * 100],
        })
        fig = px.bar(
            channel_data, x="Channel", y="Rate", text="Replies",
            color_discrete_sequence=[ACCENT_DARK],
        )
        fig.update_traces(
            texttemplate="%{text} respuestas", textposition="auto",
            marker_color=[ACCENT_DARK, ACCENT_MID],
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=340, showlegend=False)
        fig.update_yaxes(title_text="Tasa de respuesta (%)")
        st.plotly_chart(fig, use_container_width=True)

    with r2c2:
        st.markdown('<div class="section-label">Timing</div>', unsafe_allow_html=True)
        st.markdown("### Mejor dia para contactar")
        if "date" in daily.columns:
            d = daily.copy()
            d["dow"] = d["date"].dt.day_name()
            d["dow_num"] = d["date"].dt.dayofweek
            d["total_sent"] = d["linkedin_messages_sent"] + d["email_sent"]
            d["total_replies"] = d["linkedin_replies"] + d["email_replies"]
            active = d[d["total_sent"] > 0].copy()
            active["rr"] = active["total_replies"] / active["total_sent"] * 100
            dow = active.groupby(["dow_num", "dow"])["rr"].mean().reset_index()
            dow = dow[dow["dow_num"] < 5].sort_values("dow_num")

            colors = [ACCENT_GOLD if r == dow["rr"].max() else ACCENT_DARK for r in dow["rr"]]
            fig = go.Figure(go.Bar(
                x=dow["dow"], y=dow["rr"],
                text=dow["rr"].round(1), textposition="auto",
                marker_color=colors,
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=340, showlegend=False)
            fig.update_yaxes(title_text="Tasa de respuesta (%)")
            fig.update_traces(texttemplate="%{text}%")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- Row 3: Product + Segments ---
    r3c1, r3c2 = st.columns(2)

    with r3c1:
        st.markdown('<div class="section-label">Productos</div>', unsafe_allow_html=True)
        st.markdown("### Tasa de respuesta por producto")
        if "main_product" in df.columns:
            prod = df[df["main_product"] != "Unknown"].groupby("main_product").agg(
                n=("target_replied", "count"),
                replied=("target_replied", "sum"),
            ).reset_index()
            prod["rate"] = (prod["replied"] / prod["n"] * 100).round(1)
            prod = prod.sort_values("rate", ascending=True)

            fig = go.Figure(go.Bar(
                x=prod["rate"], y=prod["main_product"], orientation="h",
                text=prod["rate"], marker_color=ACCENT_DARK,
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=340, showlegend=False)
            fig.update_xaxes(title_text="Tasa de respuesta (%)")
            fig.update_traces(texttemplate="%{text}%", textposition="auto")
            st.plotly_chart(fig, use_container_width=True)

    with r3c2:
        st.markdown('<div class="section-label">Segmentacion</div>', unsafe_allow_html=True)
        st.markdown("### Segmentos de contactos")
        cl = df.groupby("cluster").agg(
            n=("target_replied", "count"),
            rate=("target_replied", "mean"),
        ).reset_index()
        cl["rate_pct"] = (cl["rate"] * 100).round(1)

        labels = [cluster_label(c) for c in cl["cluster"]]

        fig = go.Figure(go.Bar(
            x=labels,
            y=cl["n"],
            text=cl["rate_pct"],
            marker_color=[ACCENT_GOLD if r > GLOBAL_REPLY_RATE else ACCENT_DARK
                          for r in cl["rate_pct"]],
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=370, showlegend=False)
        fig.update_xaxes(tickangle=-30, tickfont=dict(size=10))
        fig.update_yaxes(title_text="Contactos")
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    # --- Segment profiles detail ---
    st.markdown("---")
    st.markdown('<div class="section-label">Detalle de segmentos</div>', unsafe_allow_html=True)
    st.markdown("### Perfiles de segmentos")

    CLUSTER_PROFILES = [
        {"Segmento": 0, "Perfil": "SMB Microsoft FIT+",
         "Tasa respuesta": "11.9%", "N": "2,588",
         "Insight": "PYME con stack Microsoft y FIT aprobado. Grupo mas grande."},
        {"Segmento": 1, "Perfil": "Enterprise Microsoft FIT+",
         "Tasa respuesta": "16.2%", "N": "1,841",
         "Insight": "Gran empresa con Microsoft. Mayor tamano de empresa."},
        {"Segmento": 2, "Perfil": "Senior Non-FIT Responders",
         "Tasa respuesta": "17.4%", "N": "581",
         "Insight": "Alta seniority, sin FIT aprobado. Mayor tasa de respuesta."},
        {"Segmento": 3, "Perfil": "Non-Microsoft FIT+",
         "Tasa respuesta": "13.6%", "N": "977",
         "Insight": "FIT aprobado, sin stack Microsoft. Oportunidad de expansion."},
    ]
    st.dataframe(
        pd.DataFrame(CLUSTER_PROFILES),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Tasa respuesta": st.column_config.TextColumn(width="small"),
            "Insight": st.column_config.TextColumn(width="large"),
        },
    )

    st.markdown("---")

    # --- Top leads table ---
    st.markdown('<div class="section-label">Ranking de leads</div>', unsafe_allow_html=True)
    st.markdown("### Contactos con mayor score")
    n_top = st.slider("Numero de leads", 10, 100, 25, label_visibility="collapsed")
    top_leads = df.nlargest(n_top, "lead_score")[
        ["Company name", "lead_score", "cluster", "ai_SENIORITY",
         "Industry", "target_replied"]
    ].copy()
    top_leads["lead_score"] = top_leads["lead_score"].round(3)
    top_leads["cluster"] = top_leads["cluster"].map(lambda c: cluster_label(c))
    top_leads.columns = ["Empresa", "Score", "Segmento", "Seniority", "Industria", "Respondio"]
    st.dataframe(top_leads, use_container_width=True, hide_index=True)


# =====================================================
# PAGE 2: LEAD SCORER
# =====================================================
with tab_scorer:
    st.markdown('<div class="top-accent"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Herramienta de scoring</div>', unsafe_allow_html=True)
    st.title("Lead Scorer Individual")
    st.markdown(
        '<div class="lead-text">'
        "Introduce el perfil de un contacto para estimar su probabilidad de respuesta. "
        "El modelo devuelve un score, asignacion de segmento y recomendacion de canal."
        "</div>",
        unsafe_allow_html=True,
    )

    # --- Input form ---
    with st.form("lead_form"):
        st.markdown("### Perfil del contacto")
        fc1, fc2, fc3 = st.columns(3)

        with fc1:
            seniority = st.selectbox(
                "Nivel de seniority",
                options=[0, 1, 2, 3, 4, 5],
                format_func=lambda x: {
                    0: "Desconocido", 1: "Junior", 2: "Lead / Senior",
                    3: "Manager", 4: "Director", 5: "VP / C-Level"
                }[x],
                index=3,
            )
            connections = st.number_input(
                "Conexiones LinkedIn", min_value=0, max_value=30000, value=500
            )
            years_role = st.number_input(
                "Anos en puesto actual", min_value=0.0, max_value=40.0, value=3.0, step=0.5
            )

        with fc2:
            employees = st.number_input(
                "Empleados de la empresa", min_value=1, max_value=500000, value=200
            )
            year_founded = st.number_input(
                "Ano de fundacion", min_value=1800, max_value=2026, value=2005
            )
            type_of_contact = st.selectbox(
                "Tipo de contacto",
                options=[0, 2, 3, 4, 5],
                format_func=lambda x: {
                    0: "Desconocido / NULL",
                    2: "Referral",
                    3: "Influencer / Champion",
                    4: "Buyer / Buyer-Champion",
                    5: "Key Decision Maker",
                }[x],
                index=2,
            )

        with fc3:
            has_bio = st.checkbox("Tiene bio en LinkedIn", value=True)
            microsoft_flag = st.checkbox("Usa tecnologia Microsoft", value=False)
            fit_approved = st.checkbox("FIT aprobado", value=True)
            hiring = st.checkbox("Ofertas de empleo activas", value=False)
            is_enriched = st.checkbox("Contacto ya enriquecido?", value=False,
                                      help="Si el contacto tiene CONTACT REPORT generado, "
                                           "se puede usar el modelo completo para un score mas preciso.")

        st.markdown("### Indicadores de crecimiento")
        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            growth_6m = st.number_input(
                "Crecimiento plantilla 6 meses (%)", min_value=-50.0, max_value=100.0,
                value=5.0, step=1.0,
            ) / 100
        with gc2:
            growth_1y = st.number_input(
                "Crecimiento plantilla anual (%)", min_value=-50.0, max_value=200.0,
                value=8.0, step=1.0,
            ) / 100
        with gc3:
            growth_2y = st.number_input(
                "Crecimiento plantilla 2 anos (%)", min_value=-50.0, max_value=300.0,
                value=15.0, step=1.0,
            ) / 100

        st.markdown("### Datos de enriquecimiento")
        st.markdown(
            '<p style="font-size:0.85rem; color:#8B8B85; margin-top:-0.5rem">'
            "Estos campos reflejan cuanta inteligencia se ha recopilado sobre el contacto. "
            "Los contactos enriquecidos obtienen scores significativamente mas altos."
            "</p>",
            unsafe_allow_html=True,
        )
        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            enrichment_level = st.selectbox(
                "Nivel de enriquecimiento",
                options=["none", "basic", "full"],
                format_func=lambda x: {
                    "none": "Sin enriquecer",
                    "basic": "Informe basico disponible",
                    "full": "Informe completo + company report",
                }[x],
                index=0,
            )
        with ec2:
            has_momentum = st.checkbox("Tiene senales de momentum", value=False)
        with ec3:
            ms_maturity = st.selectbox(
                "Microsoft maturity",
                options=[0, 1, 2, 3, 5, 8],
                format_func=lambda x: {
                    0: "None / Unknown", 1: "Basic (Exchange)",
                    2: "M365 user", 3: "M365 + Azure",
                    5: "Advanced (multi-product)", 8: "Full stack",
                }[x],
                index=0,
            )

        submitted = st.form_submit_button("Calcular score")

    # --- Scoring ---
    if submitted:
        # Map enrichment level to NLP feature values (based on actual data profiles)
        nlp_values = {
            "none": {"nlp_contact_report_length": 0.0, "nlp_report_length": 0.0},
            "basic": {"nlp_contact_report_length": 1200.0, "nlp_report_length": 3000.0},
            "full": {"nlp_contact_report_length": 2700.0, "nlp_report_length": 8000.0},
        }
        nlp = nlp_values[enrichment_level]

        contact_features = {
            "Years in role": years_role,
            "Years in company": years_role,
            "Number of connections": float(connections),
            "Number of employees": float(employees),
            "Year founded": float(year_founded),
            "Hiring on LinkedIn": float(hiring),
            "Six months headcount growth": growth_6m,
            "Two years headcount growth": growth_2y,
            "Yearly headcount growth": growth_1y,
            "fe_seniority_ord": float(seniority),
            "fe_type_of_contact_ord": float(type_of_contact),
            "fe_fit_approved": float(fit_approved),
            "fe_fit_data_approved": 0.0,
            "fe_company_age": float(2026 - year_founded),
            "fe_log_employees": float(np.log1p(employees)),
            "fe_company_size_bucket": (
                0 if employees < 10 else
                1 if employees < 50 else
                2 if employees < 250 else
                3 if employees < 1000 else 4
            ),
            "fe_log_connections": float(np.log1p(connections)),
            "fe_headcount_momentum": 0.5 * growth_6m + 0.3 * growth_1y + 0.2 * growth_2y,
            "fe_has_email": 1.0 if is_enriched else 0.0,
            "fe_has_bio": float(has_bio),
            "fe_microsoft_flag": float(microsoft_flag),
            "fe_department_encoded": 0.08,
            "ext_ms_maturity_score": float(ms_maturity),
            "ext_has_competitor_tech": 0.0,
            "nlp_report_length": nlp["nlp_report_length"],
            "nlp_contact_report_length": nlp["nlp_contact_report_length"],
            "nlp_has_momentum": float(has_momentum),
            "nlp_urgency_score": 1.0 if has_momentum else 0.0,
            "nlp_embedding_01": 4.8,
            "nlp_embedding_02": 3.0,
            "nlp_embedding_03": 5.9,
            "nlp_topic": 0.0,
        }

        df_input = pd.DataFrame([contact_features])

        # Seleccionar modelo: contactos enriquecidos usan modelo completo, otros el robusto
        if is_enriched:
            active_model = lead_scorer
            active_features = FEATURE_COLS_COMPLETE
            active_preprocessor = preprocessor
            model_label = "Modelo completo (contacto enriquecido)"
        else:
            active_model = lead_scorer_robust if lead_scorer_robust else lead_scorer
            active_features = FEATURE_COLS_ROBUST
            active_preprocessor = preprocessor_robust if preprocessor_robust else preprocessor
            model_label = "Modelo robusto (produccion)"

        for col in active_features:
            if col not in df_input.columns:
                df_input[col] = np.nan

        X = active_preprocessor.transform(df_input[active_features])
        score = float(active_model.predict_proba(X)[:, 1][0])

        cluster_feats = clustering_bundle["features"]
        df_cluster = df_input[cluster_feats].copy()
        X_cluster = clustering_bundle["scaler"].transform(
            clustering_bundle["imputer"].transform(df_cluster)
        )
        cluster = int(clustering_bundle["kmeans"].predict(X_cluster)[0])

        if score >= 0.5:
            risk_level = "Alta prioridad"
            risk_css = "rec-card rec-card-high"
        elif score >= 0.2:
            risk_level = "Prioridad media"
            risk_css = "rec-card rec-card-mid"
        else:
            risk_level = "Baja prioridad"
            risk_css = "rec-card rec-card-low"

        # --- Resultados ---
        st.markdown("---")
        st.markdown('<div class="section-label">Resultados</div>', unsafe_allow_html=True)
        st.markdown("### Resultado del scoring")

        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Lead score", f"{score:.1%}")
        rc2.metric("Prioridad", risk_level)
        rc3.metric("Segmento", cluster_label(cluster))
        rc4.metric("Mejor canal", "LinkedIn")
        st.caption(f"Modelo utilizado: {model_label}")

        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score * 100,
            number={"suffix": "%", "font": {"family": "Georgia", "size": 40, "color": ACCENT_DARK}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": ACCENT_LIGHT},
                "bar": {"color": ACCENT_GOLD},
                "bgcolor": "#F0F0EB",
                "steps": [
                    {"range": [0, 20], "color": "#E8E8E3"},
                    {"range": [20, 50], "color": "#D4D4CF"},
                    {"range": [50, 100], "color": "#FFF3D1"},
                ],
            },
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", height=250, margin=dict(t=30, b=10),
            font=dict(family="-apple-system, BlinkMacSystemFont, sans-serif"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Comparison
        st.markdown("---")
        st.markdown('<div class="section-label">Contexto</div>', unsafe_allow_html=True)
        st.markdown("### Posicion de este contacto")

        percentile = (df["lead_score"] < score).mean() * 100

        cc1, cc2 = st.columns([1, 1])
        with cc1:
            if score >= 0.5:
                rec_text = (
                    "Este contacto esta en el **top tier**. Priorizar outreach inmediato "
                    "por LinkedIn. Mejor dia: jueves. El modelo estima alta probabilidad "
                    "de respuesta."
                )
            elif score >= 0.2:
                rec_text = (
                    "Este contacto muestra **potencial moderado**. Considerar enriquecer el perfil "
                    "antes del outreach para mejorar la personalizacion. LinkedIn sigue siendo el canal preferido."
                )
            else:
                rec_text = (
                    "Este contacto tiene una **probabilidad de respuesta baja**. Evaluar si los recursos "
                    "de outreach se aprovecharian mejor en leads con mayor score."
                )

            st.markdown(
                f'<div class="{risk_css}">'
                f'<strong style="font-size:0.9rem">{risk_level}</strong><br>'
                f'<span style="font-size:0.85rem; color:#3D3D3A">Score superior al {percentile:.0f}% de todos los contactos</span>'
                f'<p style="margin-top:0.8rem; font-size:0.9rem">{rec_text}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with cc2:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df["lead_score"], nbinsx=50,
                marker_color=ACCENT_LIGHT, opacity=0.7, name="Todos los contactos",
            ))
            fig.add_vline(x=score, line_dash="dash", line_color=ACCENT_GOLD, line_width=2,
                          annotation_text=f"Este contacto ({score:.0%})",
                          annotation_font_size=11)
            fig.update_layout(**PLOTLY_LAYOUT, height=260, showlegend=False)
            fig.update_xaxes(title_text="Lead score")
            fig.update_yaxes(title_text="Cantidad")
            st.plotly_chart(fig, use_container_width=True)

        # Cluster profile
        st.markdown("---")
        st.markdown('<div class="section-label">Perfil del segmento</div>', unsafe_allow_html=True)
        cl_name = cluster_label(cluster)
        st.markdown(f"### Segmento {cluster}: {cl_name}")

        cluster_data = df[df["cluster"] == cluster]
        cluster_reply = cluster_data["target_replied"].mean() * 100
        cluster_size = len(cluster_data)
        delta_vs_global = cluster_reply - GLOBAL_REPLY_RATE

        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("Contactos en segmento", f"{cluster_size:,}")
        pc2.metric("Tasa respuesta segmento", f"{cluster_reply:.1f}%")
        pc3.metric("vs media global", f"{delta_vs_global:+.1f}pp")

        if "Industry" in cluster_data.columns:
            top_ind = (
                cluster_data["Industry"].dropna().value_counts().head(5).reset_index()
            )
            top_ind.columns = ["Industry", "Contacts"]
            st.dataframe(top_ind, use_container_width=True, hide_index=True)


# =====================================================
# PAGE 3: BATCH SCORER
# =====================================================
with tab_batch:
    st.markdown('<div class="top-accent"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Procesamiento masivo</div>', unsafe_allow_html=True)
    st.title("Batch Lead Scorer")
    st.markdown(
        '<div class="lead-text">'
        "Sube un CSV con datos de contactos para scorear todos a la vez. "
        "Se utiliza el modelo robusto (sin features de enriquecimiento) para contactos nuevos."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p style="font-size:0.9rem; color:#6B6B66;">'
        "Sube el CSV de contactos no contactados (con features ya procesadas) "
        "o cualquier CSV con las columnas del modelo."
        "</p>",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Sube tu CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"{len(batch_df)} contactos cargados")

            # Use robust model for batch
            active_features = FEATURE_COLS_ROBUST
            active_prep = preprocessor_robust if preprocessor_robust else preprocessor
            active_model_batch = lead_scorer_robust if lead_scorer_robust else lead_scorer

            # Check if CSV already has model features (pre-processed dataset)
            has_model_features = all(c in batch_df.columns for c in active_features)

            if has_model_features:
                features_df = batch_df[active_features].copy()
            else:
                st.error("El CSV no contiene las features del modelo. "
                         "Usa el CSV de contactos no contactados exportado desde el pipeline.")
                st.stop()

            for col in active_features:
                if col not in features_df.columns:
                    features_df[col] = np.nan

            X_batch = active_prep.transform(features_df[active_features])
            scores = active_model_batch.predict_proba(X_batch)[:, 1]

            # Assign clusters
            cluster_feats = clustering_bundle["features"]
            cluster_data = features_df.reindex(columns=cluster_feats, fill_value=0)
            X_cl = clustering_bundle["scaler"].transform(
                clustering_bundle["imputer"].transform(cluster_data)
            )
            clusters = clustering_bundle["kmeans"].predict(X_cl)

            # Build results
            results = batch_df.copy()
            results["lead_score"] = np.round(scores, 4)
            results["cluster"] = clusters
            results["cluster_name"] = [cluster_label(c) for c in clusters]
            results["priority"] = pd.cut(
                scores, bins=[-0.01, 0.1, 0.3, 1.01],
                labels=["Low", "Medium", "High"],
            )
            results = results.sort_values("lead_score", ascending=False)

            # Summary stats
            st.markdown("---")
            st.markdown("### Resumen de resultados")
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Total scoreados", f"{len(results):,}")
            sc2.metric("Alta prioridad", f"{(results['priority'] == 'High').sum()}")
            sc3.metric("Prioridad media", f"{(results['priority'] == 'Medium').sum()}")
            sc4.metric("Score medio", f"{results['lead_score'].mean():.1%}")

            # Score distribution
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=results["lead_score"], nbinsx=30,
                marker_color=ACCENT_GOLD, opacity=0.9,
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=False)
            fig.update_xaxes(title_text="Lead score")
            fig.update_yaxes(title_text="Cantidad")
            st.plotly_chart(fig, use_container_width=True)

            # Ranked table
            st.markdown("### Leads ordenados por score")
            display_cols = ["lead_score", "priority", "cluster_name"]
            # Add original columns that exist
            for col in ["Company name", "Industry", "Job title", "LinkedIn profile ID"]:
                if col in results.columns:
                    display_cols.insert(0, col)
            st.dataframe(
                results[display_cols].reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )

            # Download results
            from io import BytesIO
            excel_buffer = BytesIO()
            results[display_cols].to_excel(excel_buffer, index=False, engine="openpyxl")
            excel_buffer.seek(0)
            st.download_button(
                label="Descargar resultados (Excel)",
                data=excel_buffer,
                file_name="batch_scored_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        except Exception as e:
            st.error(f"Error procesando el archivo: {str(e)}")
