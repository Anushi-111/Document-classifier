import re

def preprocess_text_for_classification(text: str) -> str:
    """
    Cleans text for input into a machine learning classifier.
    THIS FUNCTION MUST BE IDENTICAL TO THE ONE USED FOR TRAINING.
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