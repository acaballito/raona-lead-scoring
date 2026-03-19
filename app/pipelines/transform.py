"""Transform: feature engineering para datos nuevos.

Logica extraida de NB03. Aplica las mismas transformaciones que el notebook
a datos nuevos para que sean compatibles con el modelo (39 features).

Requiere: scikit-learn==1.6.1, lightgbm==4.6.0
"""
import re
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

SENIORITY_MAP = {"CLEVEL": 4, "DIRECTOR": 3, "MANAGER": 2, "LEAD": 1, "JR": 0}

TYPE_OF_CONTACT_MAP = {
    "KEY_DECISION_MAKER": 5, "BUYER_CHAMPION": 4, "CHAMPION": 3,
    "INFLUENCER": 3, "REFERRAL": 2, "NULL": 0,
}

URGENCY_KEYWORDS = [
    "crecimiento", "expansion", "transformacion digital",
    "nuevo proyecto", "inversion", "presupuesto",
    "contratando", "hiring", "growth",
    "licitacion", "concurso",
    "migracion", "modernizacion",
    "urgente", "inmediato", "prioridad",
]

HR_KEYWORDS = ["HR", "RRHH", "PEOPLE", "RECURSOS HUMANOS", "HUMAN RESOURCES", "TALENT"]
COMMS_KEYWORDS = ["COMUNICACION", "COMMUNICATIONS", "MARKETING", "BRAND"]

# Scores de madurez Microsoft (extraido de NB03)
TECH_SCORES = {
    # Microsoft ecosystem (positivo)
    "azure": 3, "microsoft azure": 3,
    "dynamics 365": 3, "dynamics": 3,
    "microsoft 365": 2, "office 365": 2, "microsoft office": 2,
    "power bi": 2, "power platform": 2, "power apps": 2, "power automate": 2,
    "sharepoint": 2,
    "teams": 1, "microsoft teams": 1,
    "active directory": 1, "entra id": 1, "azure ad": 1,
    "exchange": 1, "exchange online": 1,
    "windows server": 1, "sql server": 1,
    "intune": 2, "endpoint manager": 2,
    "copilot": 2,
    # Competidores (negativo)
    "google workspace": -1, "google cloud": -1, "gcp": -1,
    "aws": -1, "amazon web services": -1,
    "salesforce": -1, "hubspot": -1,
    "slack": -1, "zoom": -1,
}

# Mapeo de productos Raona a keywords tecnologicas (extraido de NB03)
PRODUCT_TECH_MAP = {
    "comunica": ["teams", "microsoft teams", "skype"],
    "colabora": ["sharepoint", "onedrive", "power apps", "power automate", "power platform"],
    "infra": ["azure", "microsoft azure", "windows server", "sql server"],
    "ia": ["copilot", "azure ai", "cognitive", "openai"],
    "data": ["power bi", "fabric", "synapse", "sql server"],
    "workplace": ["microsoft 365", "office 365", "intune", "endpoint manager"],
    "maite": ["entra id", "azure ad", "active directory", "purview", "intune"],
}


def clean_type_of_contact(val):
    """Limpia el campo TYPE OF CONTACT a categorias estandar."""
    if pd.isna(val):
        return "NULL"
    val_clean = re.sub(r"[^a-zA-Z\s-]", "", str(val)).strip().upper()
    if "DECISOR" in val_clean:
        return "KEY_DECISION_MAKER"
    if "CHAMPION" in val_clean and "BUYER" in val_clean:
        return "BUYER_CHAMPION"
    if "CHAMPION" in val_clean:
        return "CHAMPION"
    if "INFLUENCER" in val_clean:
        return "INFLUENCER"
    if "REFERER" in val_clean or "REFERIDOR" in val_clean:
        return "REFERRAL"
    return "NULL"


def correct_department(row):
    """Corrige departamento usando Job title."""
    dept = row.get("ai_DEPARTMENT", np.nan)
    title = str(row.get("Job title", "")).upper()
    if any(kw in title for kw in HR_KEYWORDS):
        return "HR"
    if any(kw in title for kw in COMMS_KEYWORDS):
        return "COMMUNICATIONS"
    return dept


def count_urgency_keywords(text):
    """Cuenta palabras clave de urgencia en texto."""
    if pd.isna(text):
        return 0
    text_lower = str(text).lower()
    return sum(1 for kw in URGENCY_KEYWORDS if kw in text_lower)


def score_tech_stack(tech_string):
    """Calcula score de madurez Microsoft y presencia de competidores."""
    if pd.isna(tech_string):
        return 0, 0
    tech_lower = str(tech_string).lower()
    ms_score = 0
    has_competitor = 0
    for tech, score in TECH_SCORES.items():
        if tech in tech_lower:
            if score > 0:
                ms_score += score
            else:
                has_competitor = 1
    return ms_score, has_competitor


def tech_contains(tech_str, keywords):
    """Verifica si el string de tecnologias contiene alguna keyword."""
    if pd.isna(tech_str):
        return 0
    tech_lower = str(tech_str).lower()
    return int(any(kw in tech_lower for kw in keywords))


def run(df: pd.DataFrame, global_mean: float = 0.06) -> pd.DataFrame:
    """Aplica feature engineering completo a un DataFrame (39 features)."""
    logger.info(f"Feature engineering en {len(df):,} filas")

    # --- Codificaciones ordinales ---
    if "ai_SENIORITY" in df.columns:
        df["fe_seniority_ord"] = df["ai_SENIORITY"].map(SENIORITY_MAP)

    if "ai_TYPE_OF_CONTACT" in df.columns:
        df["ai_TYPE_OF_CONTACT_clean"] = df["ai_TYPE_OF_CONTACT"].apply(clean_type_of_contact)
        df["fe_type_of_contact_ord"] = df["ai_TYPE_OF_CONTACT_clean"].map(TYPE_OF_CONTACT_MAP)

    # --- FIT ---
    if "ai_FIT" in df.columns:
        df["fe_fit_approved"] = df["ai_FIT"].apply(
            lambda x: 1.0 if pd.notna(x) and "APROBADO" in str(x).upper()
                       and "DESAPROBADO" not in str(x).upper() else 0.0
        )
    if "ai_FIT_DATA" in df.columns:
        df["fe_fit_data_approved"] = df["ai_FIT_DATA"].map({"SI": 1, "NO": 0, "COMPETITOR": 0, "DUDA": 0})

    # --- Numericas ---
    if "Year founded" in df.columns:
        df["fe_company_age"] = 2026 - df["Year founded"]
    if "Number of employees" in df.columns:
        df["fe_log_employees"] = np.log1p(df["Number of employees"])
        bins = [0, 10, 50, 250, 1000, np.inf]
        labels = [0, 1, 2, 3, 4]
        df["fe_company_size_bucket"] = pd.cut(
            df["Number of employees"], bins=bins, labels=labels, right=False
        ).astype(float)
    if "Number of connections" in df.columns:
        df["fe_log_connections"] = np.log1p(df["Number of connections"])

    # --- Headcount momentum ---
    for col in ["Six months headcount growth", "Yearly headcount growth", "Two years headcount growth"]:
        if col not in df.columns:
            df[col] = 0.0
    df["fe_headcount_momentum"] = (
        0.5 * df["Six months headcount growth"].fillna(0) +
        0.3 * df["Yearly headcount growth"].fillna(0) +
        0.2 * df["Two years headcount growth"].fillna(0)
    )

    # --- Binarias ---
    if "Professional email" in df.columns:
        df["fe_has_email"] = df["Professional email"].notna().astype(int)
    if "Profile bio" in df.columns:
        df["fe_has_bio"] = df["Profile bio"].notna().astype(int)
    if "ai_Microsoft" in df.columns:
        df["fe_microsoft_flag"] = (df["ai_Microsoft"] == 1).astype(int)

    # --- Department encoding ---
    if "ai_DEPARTMENT" in df.columns:
        df["fe_department_corrected"] = df.apply(correct_department, axis=1)
        df["fe_department_encoded"] = global_mean  # Fallback con media global

    # --- Enrichment externo (ext_) ---
    if "Technologies used" in df.columns:
        tech_results = df["Technologies used"].apply(score_tech_stack)
        df["ext_ms_maturity_score"] = tech_results.apply(lambda x: x[0])
        df["ext_has_competitor_tech"] = tech_results.apply(lambda x: x[1])
    else:
        df["ext_ms_maturity_score"] = 0
        df["ext_has_competitor_tech"] = 0

    # --- Tech fit por producto (fe_tech_fit_*) ---
    if "Technologies used" in df.columns:
        for product, keywords in PRODUCT_TECH_MAP.items():
            col_name = f"fe_tech_fit_{product}"
            df[col_name] = df["Technologies used"].apply(lambda x, kw=keywords: tech_contains(x, kw))
    else:
        for product in PRODUCT_TECH_MAP:
            df[f"fe_tech_fit_{product}"] = 0

    # --- NLP basicas ---
    if "ai_COMPANY_REPORT" in df.columns:
        df["nlp_report_length"] = df["ai_COMPANY_REPORT"].str.len().fillna(0)
    else:
        df["nlp_report_length"] = 0
    if "ai_CONTACT_REPORT" in df.columns:
        df["nlp_contact_report_length"] = df["ai_CONTACT_REPORT"].str.len().fillna(0)
    else:
        df["nlp_contact_report_length"] = 0
    if "ai_MOMENTUM" in df.columns:
        df["nlp_has_momentum"] = df["ai_MOMENTUM"].notna().astype(int)
        urgency_m = df["ai_MOMENTUM"].apply(count_urgency_keywords)
        urgency_r = df.get("ai_COMPANY_REPORT", pd.Series(dtype=str)).apply(count_urgency_keywords)
        df["nlp_urgency_score"] = pd.concat([urgency_m, urgency_r], axis=1).max(axis=1)
    else:
        df["nlp_has_momentum"] = 0
        df["nlp_urgency_score"] = 0

    # --- NLP embeddings (requeririan modelo de embeddings en produccion) ---
    for col in ["nlp_embedding_01", "nlp_embedding_02", "nlp_embedding_03", "nlp_topic"]:
        if col not in df.columns:
            df[col] = 0.0

    n_features = len([c for c in df.columns if c.startswith(("fe_", "nlp_", "ext_"))])
    logger.info(f"Features creadas: {n_features} (fe_ + nlp_ + ext_)")
    return df
