"""Ingest: carga y validacion de datos nuevos.

Logica extraida de NB01. Carga un CSV de contactos, valida el esquema
y guarda en formato Parquet.
"""
import os
import re
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

REQUIRED_COLS = [
    "LinkedIn profile ID", "Company name", "Campaign engagement status",
    "Job title", "Number of employees", "Industry",
]


def load_contacts(csv_path: str) -> pd.DataFrame:
    """Carga contacts CSV y aplica limpieza basica (NB01 logic)."""
    logger.info(f"Cargando {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    logger.info(f"Filas cargadas: {len(df):,}")
    return df


def validate_schema(df: pd.DataFrame) -> bool:
    """Valida que el DataFrame tenga las columnas minimas requeridas."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        logger.error(f"Columnas faltantes: {missing}")
        return False
    logger.info("Esquema validado correctamente")
    return True


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Crea variable objetivo target_replied desde Campaign engagement status."""
    df["target_replied"] = df["Campaign engagement status"].str.contains(
        "Replied", case=False, na=False
    ).astype(int)
    logger.info(f"Target creado: {df['target_replied'].sum()} positivos ({df['target_replied'].mean():.1%})")
    return df


def filter_valid_contacts(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra contactos sin mensajes enviados y no validos (Microsoft?=-1, FIT=NO)."""
    n_before = len(df)

    # Solo contactos que recibieron al menos un mensaje
    status = df["Campaign engagement status"]
    contacted = status.str.contains("Sent|Replied|Connection Accepted", case=False, na=False)
    df = df[contacted].copy()

    # Excluir Microsoft?=-1
    if "Microsoft?" in df.columns:
        df = df[df["Microsoft?"] != -1].copy()

    # Excluir FIT=NO
    if "FIT" in df.columns:
        fit_no = df["FIT"].str.strip().str.upper() == "NO"
        df = df[~fit_no].copy()

    logger.info(f"Contactos filtrados: {n_before:,} -> {len(df):,}")
    return df


def extract_reply_message_number(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae el numero de mensaje de respuesta del status."""
    def _extract(status):
        if pd.isna(status):
            return np.nan
        match = re.search(r"Replied\s*\((\d+)\)", status)
        return int(match.group(1)) if match else np.nan

    df["reply_message_number"] = df["Campaign engagement status"].apply(_extract)
    return df


def run(csv_path: str, output_path: str) -> str:
    """Pipeline completo de ingestion."""
    df = load_contacts(csv_path)
    if not validate_schema(df):
        raise ValueError("Esquema de datos invalido")
    df = create_target(df)
    df = filter_valid_contacts(df)
    df = extract_reply_message_number(df)

    df.to_parquet(output_path, index=False)
    logger.info(f"Guardado: {output_path} ({len(df):,} filas)")
    return output_path
