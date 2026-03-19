"""Airflow DAG 1: Scoring semanal de contactos nuevos.

Pipeline: ingest >> transform >> score >> done

Se ejecuta semanalmente cuando Raona exporta nuevos contactos de Enginy.
En demo mode, divide el dataset existente 80/20 para simular datos nuevos.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

import os
import sys
import pandas as pd
import logging

logger = logging.getLogger(__name__)

BASE_DIR = os.environ.get("LEAD_SCORING_BASE_DIR", "/opt/airflow/data")
MODEL_DIR = os.environ.get("LEAD_SCORING_MODEL_DIR", "/opt/airflow/models")
RAW_DATA_DIR = os.path.join(BASE_DIR, "raw")
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


def _setup_demo_mode(**kwargs):
    """En demo mode, divide el dataset existente 80/20."""
    from sklearn.model_selection import train_test_split
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(SCORED_DIR, exist_ok=True)
    full_data_path = os.path.join(BASE_DIR, "modeling_dataset_final.parquet")
    if not os.path.exists(full_data_path):
        logger.warning(f"Dataset no encontrado: {full_data_path}")
        return
    df = pd.read_parquet(full_data_path)
    train, new = train_test_split(df, test_size=0.2, random_state=42, stratify=df["target_replied"])
    train.to_parquet(os.path.join(PROCESSED_DIR, "historical.parquet"), index=False)
    new.to_parquet(os.path.join(RAW_DATA_DIR, "new_contacts.parquet"), index=False)
    logger.info(f"Demo: {len(train)} historicos, {len(new)} nuevos")


def _ingest(**kwargs):
    """Ejecuta pipeline de ingestion."""
    input_path = os.path.join(RAW_DATA_DIR, "new_contacts.parquet")
    output_path = os.path.join(PROCESSED_DIR, "ingested.parquet")
    if input_path.endswith(".parquet"):
        df = pd.read_parquet(input_path)
        df.to_parquet(output_path, index=False)
    else:
        from pipelines import ingest
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
    """Aplica scoring."""
    from pipelines import score
    input_path = os.path.join(PROCESSED_DIR, "transformed.parquet")
    output_path = os.path.join(SCORED_DIR, f"scored_{datetime.now().strftime('%Y%m%d')}.parquet")
    score.run(input_path, MODEL_DIR, output_path)
    return output_path


with DAG(
    dag_id="raona_scoring_weekly",
    default_args=default_args,
    description="Scoring semanal: ingest -> transform -> score",
    schedule_interval="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["raona", "ml", "scoring"],
) as dag:

    setup = PythonOperator(task_id="setup_demo", python_callable=_setup_demo_mode)
    ingest = PythonOperator(task_id="ingest", python_callable=_ingest)
    transform = PythonOperator(task_id="transform", python_callable=_transform)
    score = PythonOperator(task_id="score", python_callable=_score)
    done = EmptyOperator(task_id="done")

    setup >> ingest >> transform >> score >> done
