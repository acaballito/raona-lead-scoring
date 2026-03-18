"""FastAPI endpoint para el lead scoring de Raona.

Uso:
    uvicorn api:app --host 0.0.0.0 --port 8000

Documentacion interactiva:
    http://localhost:8000/docs
"""
import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# --- Cargar modelo al iniciar ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

with open(os.path.join(MODEL_DIR, "preprocessor.pkl"), "rb") as f:
    preprocessor = pickle.load(f)
with open(os.path.join(MODEL_DIR, "lead_scorer.pkl"), "rb") as f:
    lead_scorer = pickle.load(f)
with open(os.path.join(MODEL_DIR, "clustering.pkl"), "rb") as f:
    clustering_bundle = pickle.load(f)
with open(os.path.join(MODEL_DIR, "feature_names.pkl"), "rb") as f:
    feature_names = pickle.load(f)

FEATURE_COLS = feature_names
CLUSTER_FEATURES = clustering_bundle["features"]


# --- Schemas ---
class ContactInput(BaseModel):
    """Datos de entrada del contacto."""
    years_in_company: Optional[float] = Field(None, description="Anos en la empresa")
    number_of_connections: Optional[float] = Field(None, description="Conexiones en LinkedIn")
    number_of_employees: Optional[float] = Field(None, description="Empleados de la empresa")
    year_founded: Optional[float] = Field(None, description="Ano de fundacion")
    hiring_on_linkedin: Optional[float] = Field(None, description="1 si tiene ofertas activas")
    six_months_growth: Optional[float] = Field(None, description="Crecimiento plantilla 6 meses")
    two_years_growth: Optional[float] = Field(None, description="Crecimiento plantilla 2 anos")
    yearly_growth: Optional[float] = Field(None, description="Crecimiento plantilla anual")
    fe_seniority_ord: Optional[float] = Field(None, description="Seniority ordinal (0-5)")
    fe_type_of_contact_ord: Optional[float] = Field(None, description="Tipo contacto ordinal (0-5)")
    fe_fit_approved: Optional[float] = Field(None, description="FIT aprobado (0/1)")
    fe_fit_data_approved: Optional[float] = Field(None, description="FIT DATA aprobado (0/1)")
    fe_company_age: Optional[float] = Field(None, description="Edad de la empresa")
    fe_log_employees: Optional[float] = Field(None, description="log1p(empleados)")
    fe_company_size_bucket: Optional[float] = Field(None, description="Bucket tamano (0-4)")
    fe_log_connections: Optional[float] = Field(None, description="log1p(conexiones)")
    fe_headcount_momentum: Optional[float] = Field(None, description="Momentum crecimiento")
    fe_has_bio: Optional[float] = Field(None, description="Tiene bio en LinkedIn (0/1)")
    fe_microsoft_flag: Optional[float] = Field(None, description="Usa Microsoft (0/1)")
    fe_department_encoded: Optional[float] = Field(None, description="Departamento target-encoded")
    ext_ms_maturity_score: Optional[float] = Field(None, description="Score madurez Microsoft")
    ext_has_competitor_tech: Optional[float] = Field(None, description="Usa tech competidora (0/1)")
    nlp_report_length: Optional[float] = Field(None, description="Longitud company report")
    nlp_has_momentum: Optional[float] = Field(None, description="Tiene info momentum (0/1)")
    nlp_urgency_score: Optional[float] = Field(None, description="Score de urgencia")
    nlp_embedding_01: Optional[float] = Field(None, description="Embedding UMAP dim 1")
    nlp_embedding_02: Optional[float] = Field(None, description="Embedding UMAP dim 2")
    nlp_embedding_03: Optional[float] = Field(None, description="Embedding UMAP dim 3")
    nlp_topic: Optional[float] = Field(None, description="Topic cluster asignado")


class ScoreOutput(BaseModel):
    """Resultado del scoring."""
    lead_score: float = Field(description="Probabilidad de respuesta (0-1)")
    cluster: int = Field(description="Segmento asignado (0-3)")
    risk_level: str = Field(description="ALTO (>0.5), MEDIO (0.2-0.5), BAJO (<0.2)")
    recommended_channel: str = Field(description="Canal recomendado")
    recommended_day: str = Field(description="Mejor dia para contactar")


# --- App ---
app = FastAPI(
    title="Raona Lead Scoring API",
    description="Scoring de leads B2B con modelo LightGBM",
    version="2.0.0",
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": "lead_scorer",
        "n_features": len(FEATURE_COLS),
        "n_clusters": clustering_bundle["kmeans"].n_clusters,
    }


@app.post("/score", response_model=ScoreOutput)
def score_contact(contact: ContactInput):
    try:
        field_mapping = {
            "years_in_company": "Years in company",
            "number_of_connections": "Number of connections",
            "number_of_employees": "Number of employees",
            "year_founded": "Year founded",
            "hiring_on_linkedin": "Hiring on LinkedIn",
            "six_months_growth": "Six months headcount growth",
            "two_years_growth": "Two years headcount growth",
            "yearly_growth": "Yearly headcount growth",
        }
        contact_dict = {}
        for field_name, value in contact.dict().items():
            col_name = field_mapping.get(field_name, field_name)
            contact_dict[col_name] = value

        df_input = pd.DataFrame([contact_dict])
        for col in FEATURE_COLS:
            if col not in df_input.columns:
                df_input[col] = np.nan

        X = preprocessor.transform(df_input[FEATURE_COLS])
        score = float(lead_scorer.predict_proba(X)[:, 1][0])

        df_cluster = df_input[CLUSTER_FEATURES].copy()
        for col in CLUSTER_FEATURES:
            if col not in df_cluster.columns:
                df_cluster[col] = np.nan
        X_cluster = clustering_bundle["scaler"].transform(
            clustering_bundle["imputer"].transform(df_cluster)
        )
        cluster = int(clustering_bundle["kmeans"].predict(X_cluster)[0])

        risk_level = "ALTO" if score >= 0.5 else "MEDIO" if score >= 0.2 else "BAJO"

        return ScoreOutput(
            lead_score=round(score, 4), cluster=cluster, risk_level=risk_level,
            recommended_channel="LinkedIn", recommended_day="Jueves",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
