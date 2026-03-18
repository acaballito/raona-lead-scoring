"""Retrain: reentrenamiento del modelo con datos acumulados.

Solo se ejecuta si el monitor detecta drift significativo.
Reentrena el modelo con datos acumulados.
"""
import os
import pickle
import numpy as np
import pandas as pd
import logging

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

logger = logging.getLogger(__name__)

SEED = 42


def run(data_path: str, model_dir: str, feature_names_path: str) -> dict:
    """Reentrena el modelo con los datos acumulados."""
    try:
        import lightgbm as lgb
    except ImportError:
        logger.error("LightGBM no disponible")
        return {"status": "error", "message": "LightGBM not installed"}

    # Cargar datos
    df = pd.read_parquet(data_path)
    logger.info(f"Datos para reentrenamiento: {len(df):,} filas")

    # Cargar feature names
    with open(feature_names_path, "rb") as f:
        feature_names = pickle.load(f)

    features = feature_names

    X = df[features].copy()
    y = df["target_replied"].copy()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # Preprocessor
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    # Entrenar nuevo modelo
    neg_pos_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        scale_pos_weight=neg_pos_ratio,
        random_state=SEED,
        verbose=-1,
    )
    model.fit(X_train_p, y_train)

    # Evaluar
    y_proba = model.predict_proba(X_test_p)[:, 1]
    pr_auc = average_precision_score(y_test, y_proba)
    logger.info(f"Nuevo modelo PR-AUC: {pr_auc:.4f}")

    # Guardar artefactos temporales (no reemplaza produccion hasta validacion)
    candidate_dir = os.path.join(model_dir, "candidate")
    os.makedirs(candidate_dir, exist_ok=True)

    with open(os.path.join(candidate_dir, "lead_scorer.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(candidate_dir, "preprocessor.pkl"), "wb") as f:
        pickle.dump(preprocessor, f)

    return {
        "status": "success",
        "pr_auc": pr_auc,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "candidate_dir": candidate_dir,
    }
