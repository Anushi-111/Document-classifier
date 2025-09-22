import os
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --- Configuration: These MUST match the constants in backend/config.py ---
SAVED_MODELS_DIR = "savedModelsT"
VECTORIZER_NAME = "vectorizerT.pkl"
CLASSIFIER_NAME = "classifierT.pkl"

LABEL_REPORT = "report"
LABEL_PRESCRIPTION = "prescription"
LABEL_IRRELEVANT = "irrelevant"

# --- Shared Preprocessing Function ---
# THIS FUNCTION MUST BE IDENTICAL TO THE ONE IN backend/classification_preprocessor.py
def preprocess_text_for_classification(text: str) -> str:
    """
    Cleans text for input into a machine learning classifier.
    - Converts to lowercase
    - Removes all characters except letters and spaces
    - Normalizes whitespace
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) # Keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Training Data (using the correct labels) ---
data = [
    # --- Prescriptions ---
    ("Paracetamol 500mg, take one tablet twice daily", LABEL_PRESCRIPTION),
    ("Amoxicillin 250mg capsule, 3 times a day", LABEL_PRESCRIPTION),
    ("Ibuprofen 400mg as needed for pain", LABEL_PRESCRIPTION),
    ("Metformin 500mg, once daily after meals", LABEL_PRESCRIPTION),
    ("Cough syrup 10ml, morning and night", LABEL_PRESCRIPTION),

    # --- Lab Reports ---
    ("Hemoglobin: 13.5 g/dL, WBC: 6000, Platelets: 2.1 lakh", LABEL_REPORT),
    ("Blood sugar fasting: 92 mg/dL, postprandial: 110 mg/dL", LABEL_REPORT),
    ("X-Ray chest: No abnormalities detected", LABEL_REPORT),
    ("MRI Brain: Normal scan, no lesion found", LABEL_REPORT),
    ("Urine test: Protein negative, Sugar negative", LABEL_REPORT),

    # --- Irrelevant / Invalid Documents ---
    ("Shopping bill for groceries", LABEL_IRRELEVANT),
    ("Flight ticket from Delhi to Mumbai", LABEL_IRRELEVANT),
    ("Bank statement for January", LABEL_IRRELEVANT),
    ("Movie ticket: Avengers Endgame", LABEL_IRRELEVANT),
    ("Electricity bill for March", LABEL_IRRELEVANT),
]

# --- Preprocess and Prepare Data ---
# Apply the exact same cleaning function that the backend will use
texts = [preprocess_text_for_classification(x[0]) for x in data]
labels = [x[1] for x in data]

print("--- Sample of Preprocessed Text for Training ---")
print(f"Original: {data[5][0]}")
print(f"Cleaned:  {texts[5]}")
print("-" * 45)

# --- Train Model ---
print("Training TF-IDF Vectorizer...")
# Using stop_words='english' is a good practice to remove common words
vectorizer = TfidfVectorizer(stop_words='english')
xVec = vectorizer.fit_transform(texts)

print("Training Logistic Regression Classifier...")
classifier = LogisticRegression(max_iter=1000)
classifier.fit(xVec, labels)

# --- Save Model and Vectorizer ---
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
vectorizer_path = os.path.join(SAVED_MODELS_DIR, VECTORIZER_NAME)
classifier_path = os.path.join(SAVED_MODELS_DIR, CLASSIFIER_NAME)

joblib.dump(vectorizer, vectorizer_path)
joblib.dump(classifier, classifier_path)

print(f"Vectorizer saved to: {vectorizer_path}")
print(f"Classifier saved to: {classifier_path}")
print("\nModel training and saving complete!")