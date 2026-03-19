"""Airflow DAG 2: Monitoring mensual + reentrenamiento condicional.

Pipeline: monitor >> check_drift >> [retrain >> validate | skip_retrain] >> done

Se ejecuta mensualmente. Si detecta drift significativo (PSI > 0.25 en 3+ features),
dispara reentrenamiento automatico y valida el modelo candidato.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

import os
import sys
import pickle
import pandas as pd
import logging

logger = logging.getLogger(__name__)

BASE_DIR = os.environ.get("LEAD_SCORING_BASE_DIR", "/opt/airflow/data")
MODEL_DIR = os.environ.get("LEAD_SCORING_MODEL_DIR", "/opt/airflow/models")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
SCORED_DIR = os.path.join(BASE_DIR, "scored")

PIPELINES_DIR = os.environ.get("PIPELINES_DIR", "/opt/airflow/pipelines")
if PIPELINES_DIR not in sys.path:
    sys.path.insert(0, PIPELINES_DIR)

default_args = {
    "owner": "raona-data",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _monitor(**kwargs):
    """Calcula PSI y decide si reentrenar."""
    from pipelines import monitor
    train_path = os.path.join(PROCESSED_DIR, "historical.parquet")
    new_path = os.path.join(PROCESSED_DIR, "transformed.parquet")
    feature_names_path = os.path.join(MODEL_DIR, "feature_names.pkl")
    with open(feature_names_path, "rb") as f:
        features = pickle.load(f)
    psi_df = monitor.run(train_path, new_path, features)
    psi_path = os.path.join(SCORED_DIR, f"psi_{datetime.now().strftime('%Y%m%d')}.csv")
    psi_df.to_csv(psi_path, index=False)
    n_alerts = (psi_df["status"] == "ALERTA").sum()
    kwargs["ti"].xcom_push(key="n_drift_alerts", value=n_alerts)
    return n_alerts


def _check_drift(**kwargs):
    """Branch: reentrenar solo si hay drift significativo."""
    n_alerts = kwargs["ti"].xcom_pull(task_ids="monitor", key="n_drift_alerts")
    if n_alerts and n_alerts > 0:
        return "retrain"
    return "skip_retrain"


def _retrain(**kwargs):
    """Reentrena modelo con datos acumulados."""
    from pipelines import retrain
    hist_path = os.path.join(PROCESSED_DIR, "historical.parquet")
    new_path = os.path.join(PROCESSED_DIR, "transformed.parquet")
    hist = pd.read_parquet(hist_path)
    new = pd.read_parquet(new_path)
    combined = pd.concat([hist, new], ignore_index=True)
    combined_path = os.path.join(PROCESSED_DIR, "combined_for_retrain.parquet")
    combined.to_parquet(combined_path, index=False)
    feature_names_path = os.path.join(MODEL_DIR, "feature_names.pkl")
    result = retrain.run(combined_path, MODEL_DIR, feature_names_path)
    kwargs["ti"].xcom_push(key="retrain_result", value=result)
    return result


def _validate(**kwargs):
    """Valida modelo candidato vs actual."""
    from pipelines import validate
    retrain_result = kwargs["ti"].xcom_pull(task_ids="retrain", key="retrain_result")
    if not retrain_result or retrain_result.get("status") != "success":
        logger.warning("Reentrenamiento fallido, saltando validacion")
        return
    holdout_path = os.path.join(PROCESSED_DIR, "transformed.parquet")
    feature_names_path = os.path.join(MODEL_DIR, "feature_names.pkl")
    result = validate.run(holdout_path, MODEL_DIR, retrain_result["candidate_dir"], feature_names_path)
    return result


with DAG(
    dag_id="raona_monitoring_monthly",
    default_args=default_args,
    description="Monitoring mensual: PSI drift -> reentrenamiento condicional",
    schedule_interval="@monthly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["raona", "ml", "monitoring"],
) as dag:

    monitor = PythonOperator(task_id="monitor", python_callable=_monitor)
    check_drift = BranchPythonOperator(task_id="check_drift", python_callable=_check_drift)
    retrain = PythonOperator(task_id="retrain", python_callable=_retrain)
    validate = PythonOperator(task_id="validate", python_callable=_validate)
    skip_retrain = EmptyOperator(task_id="skip_retrain")
    done = EmptyOperator(task_id="done", trigger_rule="none_failed_min_one_success")

    monitor >> check_drift
    check_drift >> retrain >> validate >> done
    check_drift >> skip_retrain >> done
