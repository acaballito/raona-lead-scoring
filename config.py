"""Configuracion centralizada del proyecto Raona Lead Scoring.

Paths, constantes y parametros compartidos entre notebooks, app y pipelines.
"""
import os

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DATA = os.path.join(PROJECT_ROOT, "..", "raw_data")
WORKING_DIR = os.path.join(PROJECT_ROOT, "..", "_working")
WORKING_DATA = os.path.join(WORKING_DIR, "data")
WORKING_MODELS = os.path.join(WORKING_DIR, "models")
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
MLRUNS_DIR = os.path.join(WORKING_DIR, "mlruns")

APP_DIR = os.path.join(PROJECT_ROOT, "app")
DELIVERABLE_MODELS = os.path.join(APP_DIR, "models")
DELIVERABLE_DATA = os.path.join(APP_DIR, "data")
REPORT_DIR = os.path.join(PROJECT_ROOT, "report")

# --- Model parameters ---
SEED = 42
TEST_SIZE = 0.2

# --- Feature groups ---
LEAKING_FEATURES = ["nlp_contact_report_length", "fe_has_email", "Years in role"]

# --- Product lines ---
PRODUCT_LINES = ["COLABORA", "COMUNICA", "IA", "INFRA", "DATA", "WORKPLACE", "MAITE"]

# --- Type of Contact ordinal mapping ---
TYPE_OF_CONTACT_MAP = {
    "KEY_DECISION_MAKER": 5,
    "BUYER_CHAMPION": 4,
    "CHAMPION": 3,
    "INFLUENCER": 3,
    "REFERRAL": 2,
    "NULL": 0,
}

# --- Seniority ordinal mapping ---
SENIORITY_MAP = {"CLEVEL": 4, "DIRECTOR": 3, "MANAGER": 2, "LEAD": 1, "JR": 0}

# --- PSI thresholds ---
PSI_OK = 0.10
PSI_WARN = 0.25
