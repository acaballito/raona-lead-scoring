"""Monitor: detecta drift en features usando PSI.

Logica extraida de NB05. Calcula Population Stability Index entre
los datos de entrenamiento y los datos nuevos.
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Calcula Population Stability Index entre dos distribuciones."""
    breakpoints = np.percentile(expected[~np.isnan(expected)], np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 3:
        return 0.0
    expected_counts = np.histogram(expected[~np.isnan(expected)], bins=breakpoints)[0]
    actual_counts = np.histogram(actual[~np.isnan(actual)], bins=breakpoints)[0]
    expected_pct = (expected_counts + 1) / (expected_counts.sum() + len(expected_counts))
    actual_pct = (actual_counts + 1) / (actual_counts.sum() + len(actual_counts))
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def classify_psi(psi: float) -> str:
    """Clasifica el nivel de drift segun el PSI."""
    if psi < 0.1:
        return "OK"
    elif psi < 0.25:
        return "MONITORIZAR"
    return "ALERTA"


def run(train_path: str, new_path: str, features: list) -> pd.DataFrame:
    """Calcula PSI para cada feature entre datos de entrenamiento y nuevos."""
    train_df = pd.read_parquet(train_path)
    new_df = pd.read_parquet(new_path)
    results = []
    for feat in features:
        if feat in train_df.columns and feat in new_df.columns:
            train_vals = train_df[feat].values.astype(float)
            new_vals = new_df[feat].values.astype(float)
            psi = calculate_psi(train_vals, new_vals)
            status = classify_psi(psi)
            results.append({"feature": feat, "psi": round(psi, 4), "status": status})
            if status == "ALERTA":
                logger.warning(f"DRIFT DETECTADO en {feat}: PSI={psi:.4f}")
    psi_df = pd.DataFrame(results)
    logger.info(f"PSI calculado para {len(results)} features. Alertas: {(psi_df['status'] == 'ALERTA').sum()}")
    return psi_df
