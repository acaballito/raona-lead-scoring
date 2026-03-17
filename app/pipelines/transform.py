"""Transform: feature engineering para datos nuevos.

Logica extraida de NB03. Aplica las mismas transformaciones que el notebook
a datos nuevos para que sean compatibles con el modelo.
"""
import re
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

SENIORITY_MAP = {"CLEVEL": 4, "DIRECTOR": 3, "MANAGER": 2, "LEAD": 1, "JR": 0}

TYPE_OF_CONTACT_MAP = {
    "KEY_DECISION_MAKER": 5,
    "BUYER_CHAMPION": 4,
    "CHAMPION": 3,
    "INFLUENCER": 3,
    "REFERRAL": 2,
    "NULL": 0,
}

URGENCY_KEYWORDS = [
    "crecimiento", "expansion", "transformacion digital", "transformacion digital",
    "nuevo proyecto", "inversion", "inversion", "presupuesto",
    "contratando", "hiring", "growth",
    "licitacion", "licitacion", "concurso",
    "migracion", "migracion", "modernizacion", "modernizacion",
    "urgente", "inmediato", "prioridad",
]

HR_KEYWORDS = ["HR", "RRHH", "PEOPLE", "RECURSOS HUMANOS", "HUMAN RESOURCES", "TALENT"]
COMMS_KEYWORDS = ["COMUNICACION", "COMUNICACION", "COMMUNICATIONS", "MARKETING", "BRAND"]


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


def run(df: pd.DataFrame, global_mean: float = 0.06) -> pd.DataFrame:
    """Aplica feature engineering a un DataFrame."""
    logger.info(f"Feature engineering en {len(df):,} filas")

    # Codificaciones ordinales
    if "ai_SENIORITY" in df.columns:
        df["fe_seniority_ord"] = df["ai_SENIORITY"].map(SENIORITY_MAP)

    if "ai_TYPE_OF_CONTACT" in df.columns:
        df["ai_TYPE_OF_CONTACT_clean"] = df["ai_TYPE_OF_CONTACT"].apply(clean_type_of_contact)
        df["fe_type_of_contact_ord"] = df["ai_TYPE_OF_CONTACT_clean"].map(TYPE_OF_CONTACT_MAP)

    # FIT
    if "ai_FIT" in df.columns:
        df["fe_fit_approved"] = df["ai_FIT"].apply(
            lambda x: 1.0 if pd.notna(x) and "APROBADO" in str(x).upper()
                       and "DESAPROBADO" not in str(x).upper() else 0.0
        )

    if "ai_FIT_DATA" in df.columns:
        df["fe_fit_data_approved"] = df["ai_FIT_DATA"].map({"SI": 1, "NO": 0, "COMPETITOR": 0, "DUDA": 0})

    # Numericas
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

    # Headcount momentum
    for col in ["Six months headcount growth", "Yearly headcount growth", "Two years headcount growth"]:
        if col not in df.columns:
            df[col] = 0.0
    df["fe_headcount_momentum"] = (
        0.5 * df["Six months headcount growth"].fillna(0) +
        0.3 * df["Yearly headcount growth"].fillna(0) +
        0.2 * df["Two years headcount growth"].fillna(0)
    )

    # Binarias
    if "Professional email" in df.columns:
        df["fe_has_email"] = df["Professional email"].notna().astype(int)
    if "Profile bio" in df.columns:
        df["fe_has_bio"] = df["Profile bio"].notna().astype(int)
    if "ai_Microsoft" in df.columns:
        df["fe_microsoft_flag"] = (df["ai_Microsoft"] == 1).astype(int)

    # Department
    if "ai_DEPARTMENT" in df.columns:
        df["fe_department_corrected"] = df.apply(correct_department, axis=1)
        df["fe_department_encoded"] = global_mean  # Fallback

    # NLP basicas
    if "ai_COMPANY_REPORT" in df.columns:
        df["nlp_report_length"] = df["ai_COMPANY_REPORT"].str.len().fillna(0)
    if "ai_CONTACT_REPORT" in df.columns:
        df["nlp_contact_report_length"] = df["ai_CONTACT_REPORT"].str.len().fillna(0)
    if "ai_MOMENTUM" in df.columns:
        df["nlp_has_momentum"] = df["ai_MOMENTUM"].notna().astype(int)
        urgency_m = df["ai_MOMENTUM"].apply(count_urgency_keywords)
        urgency_r = df.get("ai_COMPANY_REPORT", pd.Series(dtype=str)).apply(count_urgency_keywords)
        df["nlp_urgency_score"] = pd.concat([urgency_m, urgency_r], axis=1).max(axis=1)

    # Defaults para NLP embeddings (requeririan modelo de embeddings)
    for col in ["nlp_embedding_01", "nlp_embedding_02", "nlp_embedding_03", "nlp_topic"]:
        if col not in df.columns:
            df[col] = 0.0

    logger.info(f"Features creadas: {[c for c in df.columns if c.startswith('fe_') or c.startswith('nlp_')]}")
    return df
