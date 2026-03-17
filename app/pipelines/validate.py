"""Validate: compara modelo candidato vs modelo actual.

Promueve el nuevo modelo a produccion solo si mejora el PR-AUC
en un holdout set independiente.
"""
import os
import pickle
import shutil
import numpy as np
import pandas as pd
import logging

from sklearn.metrics import average_precision_score

logger = logging.getLogger(__name__)

MIN_IMPROVEMENT = 0.01  # Mejora minima requerida en PR-AUC


def run(data_path: str, model_dir: str, candidate_dir: str, feature_names_path: str) -> dict:
    """Compara candidato vs actual y promueve si es mejor."""
    # Cargar datos holdout
    df = pd.read_parquet(data_path)
    y = df["target_replied"]

    # Feature names
    with open(feature_names_path, "rb") as f:
        feature_names = pickle.load(f)
    features = feature_names["robust"] if isinstance(feature_names, dict) else feature_names

    X = df[features].copy()

    # Modelo actual
    current_prep_path = os.path.join(model_dir, "preprocessor_robust.pkl")
    current_model_path = os.path.join(model_dir, "lead_scorer.pkl")

    if os.path.exists(current_prep_path):
        with open(current_prep_path, "rb") as f:
            current_prep = pickle.load(f)
    else:
        with open(os.path.join(model_dir, "preprocessor.pkl"), "rb") as f:
            current_prep = pickle.load(f)

    with open(current_model_path, "rb") as f:
        current_model = pickle.load(f)

    X_current = current_prep.transform(X)
    current_score = average_precision_score(y, current_model.predict_proba(X_current)[:, 1])

    # Modelo candidato
    with open(os.path.join(candidate_dir, "preprocessor_robust.pkl"), "rb") as f:
        candidate_prep = pickle.load(f)
    with open(os.path.join(candidate_dir, "lead_scorer.pkl"), "rb") as f:
        candidate_model = pickle.load(f)

    X_candidate = candidate_prep.transform(X)
    candidate_score = average_precision_score(y, candidate_model.predict_proba(X_candidate)[:, 1])

    improvement = candidate_score - current_score

    logger.info(f"PR-AUC actual:     {current_score:.4f}")
    logger.info(f"PR-AUC candidato:  {candidate_score:.4f}")
    logger.info(f"Mejora:            {improvement:+.4f}")

    result = {
        "current_pr_auc": current_score,
        "candidate_pr_auc": candidate_score,
        "improvement": improvement,
    }

    if improvement >= MIN_IMPROVEMENT:
        # Promover candidato a produccion
        shutil.copy(
            os.path.join(candidate_dir, "lead_scorer.pkl"),
            os.path.join(model_dir, "lead_scorer.pkl"),
        )
        shutil.copy(
            os.path.join(candidate_dir, "preprocessor_robust.pkl"),
            os.path.join(model_dir, "preprocessor_robust.pkl"),
        )
        result["promoted"] = True
        logger.info("Modelo candidato PROMOVIDO a produccion")
    else:
        result["promoted"] = False
        logger.info(f"Candidato NO promovido (mejora {improvement:.4f} < umbral {MIN_IMPROVEMENT})")

    # Limpiar directorio candidato
    shutil.rmtree(candidate_dir, ignore_errors=True)

    return result
