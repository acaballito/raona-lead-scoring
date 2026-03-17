"""Airflow DAG para el pipeline de Lead Scoring de Raona.

Pipeline:
    ingest >> transform >> score >> monitor >> [retrain >> validate]
                                                (condicional: solo si drift)

Demo mode: divide el dataset existente 80/20 para simular datos nuevos.
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

# Paths (configurables via Airflow Variables en produccion)
BASE_DIR = os.environ.get("LEAD_SCORING_BASE_DIR", "/opt/airflow/data")
MODEL_DIR = os.environ.get("LEAD_SCORING_MODEL_DIR", "/opt/airflow/models")
RAW_DATA_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
SCORED_DIR = os.path.join(BASE_DIR, "scored")

# Asegurar que los modulos de pipeline estan en el path
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


def _setup_demo_mode(**kwargs):
    """En demo mode, divide el dataset existente 80/20."""
    from sklearn.model_selection import train_test_split

    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(SCORED_DIR, exist_ok=True)

    # Cargar dataset completo
    full_data_path = os.path.join(BASE_DIR, "modeling_dataset_final.parquet")
    if not os.path.exists(full_data_path):
        logger.warning(f"Dataset no encontrado: {full_data_path}")
        return

    df = pd.read_parquet(full_data_path)
    train, new = train_test_split(df, test_size=0.2, random_state=42,
                                   stratify=df["target_replied"])

    train_path = os.path.join(PROCESSED_DIR, "historical.parquet")
    new_path = os.path.join(RAW_DATA_DIR, "new_contacts.parquet")
    train.to_parquet(train_path, index=False)
    new.to_parquet(new_path, index=False)
    logger.info(f"Demo: {len(train)} historicos, {len(new)} nuevos")


def _ingest(**kwargs):
    """Ejecuta pipeline de ingestion."""
    from pipelines import ingest
    input_path = os.path.join(RAW_DATA_DIR, "new_contacts.parquet")
    output_path = os.path.join(PROCESSED_DIR, "ingested.parquet")

    # Si es parquet ya procesado (demo mode), solo copiar
    if input_path.endswith(".parquet"):
        df = pd.read_parquet(input_path)
        df.to_parquet(output_path, index=False)
    else:
        ingest.run(input_path, output_path)
    return output_path


def _transform(**kwargs):
    """Ejecuta feature engineering."""
    from pipelines import transform
    input_path = os.path.join(PROCESSED_DIR, "ingested.parquet")
    df = pd.read_parquet(input_path)
    df = transform.run(df)
    output_path = os.path.join(PROCESSED_DIR, "transformed.parquet")
    df.to_parquet(output_path, index=False)
    return output_path


def _score(**kwargs):
    """Aplica scoring con modelo robusto."""
    from pipelines import score
    input_path = os.path.join(PROCESSED_DIR, "transformed.parquet")
    output_path = os.path.join(SCORED_DIR, f"scored_{datetime.now().strftime('%Y%m%d')}.parquet")
    score.run(input_path, MODEL_DIR, output_path)
    return output_path


def _monitor(**kwargs):
    """Calcula PSI y decide si reentrenar."""
    from pipelines import monitor

    train_path = os.path.join(PROCESSED_DIR, "historical.parquet")
    new_path = os.path.join(PROCESSED_DIR, "transformed.parquet")

    feature_names_path = os.path.join(MODEL_DIR, "feature_names.pkl")
    with open(feature_names_path, "rb") as f:
        feature_names = pickle.load(f)
    features = feature_names["robust"] if isinstance(feature_names, dict) else feature_names

    psi_df = monitor.run(train_path, new_path, features)

    # Guardar resultados
    psi_path = os.path.join(SCORED_DIR, f"psi_{datetime.now().strftime('%Y%m%d')}.csv")
    psi_df.to_csv(psi_path, index=False)

    # Decidir si reentrenar
    n_alerts = (psi_df["status"] == "ALERTA").sum()
    kwargs["ti"].xcom_push(key="n_drift_alerts", value=n_alerts)
    kwargs["ti"].xcom_push(key="psi_path", value=psi_path)
    return n_alerts


def _check_drift(**kwargs):
    """Branch: reentrenar solo si hay drift significativo."""
    n_alerts = kwargs["ti"].xcom_pull(task_ids="monitor", key="n_drift_alerts")
    if n_alerts and n_alerts > 0:
        return "retrain"
    return "skip_retrain"


def _retrain(**kwargs):
    """Reentrena modelo."""
    from pipelines import retrain
    # Combinar historicos + nuevos para reentrenamiento
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
    result = validate.run(
        holdout_path, MODEL_DIR, retrain_result["candidate_dir"], feature_names_path
    )
    return result


with DAG(
    dag_id="raona_lead_scoring",
    default_args=default_args,
    description="Pipeline de lead scoring: ingest -> transform -> score -> monitor -> [retrain]",
    schedule_interval="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["raona", "ml", "lead-scoring"],
) as dag:

    setup = PythonOperator(
        task_id="setup_demo",
        python_callable=_setup_demo_mode,
    )

    ingest = PythonOperator(
        task_id="ingest",
        python_callable=_ingest,
    )

    transform = PythonOperator(
        task_id="transform",
        python_callable=_transform,
    )

    score = PythonOperator(
        task_id="score",
        python_callable=_score,
    )

    monitor = PythonOperator(
        task_id="monitor",
        python_callable=_monitor,
    )

    check_drift = BranchPythonOperator(
        task_id="check_drift",
        python_callable=_check_drift,
    )

    retrain = PythonOperator(
        task_id="retrain",
        python_callable=_retrain,
    )

    validate = PythonOperator(
        task_id="validate",
        python_callable=_validate,
    )

    skip_retrain = EmptyOperator(
        task_id="skip_retrain",
    )

    done = EmptyOperator(
        task_id="done",
        trigger_rule="none_failed_min_one_success",
    )

    # DAG flow
    setup >> ingest >> transform >> score >> monitor >> check_drift
    check_drift >> retrain >> validate >> done
    check_drift >> skip_retrain >> done
