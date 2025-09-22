import joblib
from .config import VECTORIZER_PATH, CLASSIFIER_PATH
from .classification_preprocessor import preprocess_text_for_classification

# --- Load Models ---
# The paths are now imported from config.py, making them easy to change in one place.
try:
    vectorizer = joblib.load(VECTORIZER_PATH)
    classifier = joblib.load(CLASSIFIER_PATH)
    print("Classifier and vectorizer loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: Could not find model files at {VECTORIZER_PATH} or {CLASSIFIER_PATH}")
    print("Please run the `classifierModelT.py` training script first.")
    # In a real app, you might want to exit or raise an exception here
    vectorizer = None
    classifier = None

def classify_text(raw_text: str) -> str:
    """
    Takes a raw text string, preprocesses it using the shared function,
    and returns the model's classification prediction.
    """
    if not vectorizer or not classifier:
        return "Error: Models are not loaded."

    # 1. Preprocess the input text exactly as done during training
    processed_text = preprocess_text_for_classification(raw_text)

    # 2. Transform the cleaned text using the loaded vectorizer
    text_vector = vectorizer.transform([processed_text])

    # 3. Predict using the loaded classifier
    prediction = classifier.predict(text_vector)

    # The prediction is an array, so we return the first element
    return prediction[0]

