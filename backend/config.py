import os

# --- Project Root ---
# Assumes this config file is inside the 'backend' folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..")

# --- Model & Vectorizer Paths ---
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "savedModelsT")
VECTORIZER_PATH = os.path.join(SAVED_MODELS_DIR, "vectorizerT.pkl")
CLASSIFIER_PATH = os.path.join(SAVED_MODELS_DIR, "classifierT.pkl")

# --- Document Classification Labels ---
# Use these constants throughout the project to avoid typos and mismatches
LABEL_REPORT = "report"
LABEL_PRESCRIPTION = "prescription"
LABEL_IRRELEVANT = "irrelevant"

VALID_LABELS = [LABEL_REPORT, LABEL_PRESCRIPTION]

# --- Upload Directory ---
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_docs")