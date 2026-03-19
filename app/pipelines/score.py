"""Score: aplica modelo de lead scoring y clustering a datos nuevos.

Logica extraida de NB04. Usa el modelo LightGBM para scoring de contactos nuevos.

Requiere: scikit-learn==1.6.1, lightgbm==4.6.0
"""
import os
import pickle
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def load_artifacts(model_dir: str) -> dict:
    """Carga todos los artefactos del modelo."""
    artifacts = {}
    for name in ["lead_scorer", "preprocessor", "clustering", "feature_names"]:
        path = os.path.join(model_dir, f"{name}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                artifacts[name] = pickle.load(f)
            logger.info(f"Cargado: {name}")
    return artifacts


def score_leads(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Aplica scoring y asigna clusters."""
    features = artifacts["feature_names"]
    prep = artifacts["preprocessor"]
    model = artifacts["lead_scorer"]

    X = df.reindex(columns=features, fill_value=np.nan)
    X_processed = prep.transform(X)
    df["lead_score"] = model.predict_proba(X_processed)[:, 1]
    logger.info(f"Scores generados: media={df['lead_score'].mean():.3f}")

    clustering = artifacts["clustering"]
    cluster_feats = clustering["features"]
    X_cl = df.reindex(columns=cluster_feats, fill_value=0)
    X_cl_scaled = clustering["scaler"].transform(
        clustering["imputer"].transform(X_cl)
    )
    df["cluster"] = clustering["kmeans"].predict(X_cl_scaled)

    df["priority"] = pd.cut(
        df["lead_score"], bins=[-0.01, 0.1, 0.3, 1.01],
        labels=["Low", "Medium", "High"],
    )
    logger.info(f"Distribucion: {df['priority'].value_counts().to_dict()}")
    return df


def run(parquet_path: str, model_dir: str, output_path: str) -> str:
    """Pipeline de scoring."""
    df = pd.read_parquet(parquet_path)
    artifacts = load_artifacts(model_dir)
    df = score_leads(df, artifacts)
    df.to_parquet(output_path, index=False)
    logger.info(f"Guardado: {output_path} ({len(df):,} filas)")
    return output_path
