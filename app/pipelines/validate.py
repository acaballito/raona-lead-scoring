"""Validate: compara modelo candidato vs modelo actual.

Promueve el nuevo modelo a produccion solo si mejora el PR-AUC
en un holdout set independiente.

Requiere: scikit-learn==1.6.1, lightgbm==4.6.0
"""
import os
import pickle
import shutil
import numpy as np
import pandas as pd
import logging

from sklearn.metrics import average_precision_score

logger = logging.getLogger(__name__)
MIN_IMPROVEMENT = 0.01


def run(data_path: str, model_dir: str, candidate_dir: str, feature_names_path: str) -> dict:
    """Compara candidato vs actual y promueve si es mejor."""
    df = pd.read_parquet(data_path)
    y = df["target_replied"]

    with open(feature_names_path, "rb") as f:
        features = pickle.load(f)
    X = df[features].copy()

    # Modelo actual
    with open(os.path.join(model_dir, "preprocessor.pkl"), "rb") as f:
        current_prep = pickle.load(f)
    with open(os.path.join(model_dir, "lead_scorer.pkl"), "rb") as f:
        current_model = pickle.load(f)
    X_current = current_prep.transform(X)
    current_score = average_precision_score(y, current_model.predict_proba(X_current)[:, 1])

    # Modelo candidato
    with open(os.path.join(candidate_dir, "preprocessor.pkl"), "rb") as f:
        candidate_prep = pickle.load(f)
    with open(os.path.join(candidate_dir, "lead_scorer.pkl"), "rb") as f:
        candidate_model = pickle.load(f)
    X_candidate = candidate_prep.transform(X)
    candidate_score = average_precision_score(y, candidate_model.predict_proba(X_candidate)[:, 1])

    improvement = candidate_score - current_score
    logger.info(f"PR-AUC actual: {current_score:.4f}, candidato: {candidate_score:.4f}, mejora: {improvement:+.4f}")

    result = {
        "current_pr_auc": current_score,
        "candidate_pr_auc": candidate_score,
        "improvement": improvement,
    }

    if improvement >= MIN_IMPROVEMENT:
        shutil.copy(os.path.join(candidate_dir, "lead_scorer.pkl"), os.path.join(model_dir, "lead_scorer.pkl"))
        shutil.copy(os.path.join(candidate_dir, "preprocessor.pkl"), os.path.join(model_dir, "preprocessor.pkl"))
        result["promoted"] = True
        logger.info("Modelo candidato PROMOVIDO a produccion")
    else:
        result["promoted"] = False
        logger.info(f"Candidato NO promovido (mejora {improvement:.4f} < umbral {MIN_IMPROVEMENT})")

    shutil.rmtree(candidate_dir, ignore_errors=True)
    return result
