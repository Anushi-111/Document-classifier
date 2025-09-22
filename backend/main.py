import shutil
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid

# Import from your existing modules
from backend.text_extractor import extract_text_from_pdf
from backend.classification_preprocessor import preprocess_text_for_classification
from backend.classifytesting import classify_text
from backend.preprocessor import preprocess_report
from backend.config import UPLOAD_DIR, LABEL_REPORT, LABEL_PRESCRIPTION, LABEL_IRRELEVANT

# --- FastAPI App Initialization ---
app = FastAPI(title="Document Classifier API")

# --- CORS Configuration ---
# This allows your frontend to communicate with your backend.
# The "null" origin is the final fix that allows testing with a local HTML file.
origins = [
    "http://localhost",
    "http://localhost:3000",
    "null",  # <-- CRITICAL LINE FOR LOCAL FILE TESTING
    "https://document-classifier-m9yb.onrender.com" # Your live Render URL
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Directory Setup ---
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- API Endpoints ---
@app.get("/health", summary="Health Check")
async def health_check():
    """A simple endpoint to check if the server is running."""
    return {"status": "ok"}

@app.post("/upload-document/", summary="Upload and Classify Document")
async def upload_document(file: UploadFile = File(...)):
    """
    Receives a PDF file, classifies it, and if it's a report,
    extracts structured data.
    """
    temp_file_path = os.path.join(UPLOAD_DIR, f"temp_{uuid.uuid4()}_{file.filename}")

    try:
        # 1. Save the uploaded file temporarily
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"[INFO] File temporarily saved to: {temp_file_path}")

        # 2. Extract text from the PDF
        extracted_text = extract_text_from_pdf(temp_file_path)
        
        # --- Helpful debugging line to see what the server's OCR extracts ---
        print("--- EXTRACTED TEXT ---")
        print(extracted_text)
        print("--- END OF TEXT ---")
        
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract any text from the document.")

        # 3. Classify the document type
        prediction = classify_text(extracted_text)
        print(f"[INFO] Model prediction: '{prediction}'")

        # 4. Process based on classification
        if prediction == LABEL_REPORT:
            print("[INFO] Document classified as a Report. Extracting structured data...")
            report_df = preprocess_report(extracted_text)
            report_data = report_df.to_dict(orient="records")
            print(f"[ACCEPTED] Processed report with {len(report_data)} rows.")
            return JSONResponse(
                content={"status": "success", "type": prediction, "data": report_data, "message": "Report processed successfully."}
            )

        elif prediction == LABEL_PRESCRIPTION:
            print("[ACCEPTED] Document classified as a Prescription.")
            return JSONResponse(
                content={"status": "success", "type": prediction, "message": "Prescription submitted successfully."}
            )
        
        else: # The prediction is 'irrelevant'
            print(f"[REJECTED] Document classified as Irrelevant.")
            raise HTTPException(status_code=400, detail="The uploaded document is not a valid report or prescription.")

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions to be handled by FastAPI
        raise http_exc
    except Exception as e:
        print(f"[CRITICAL ERROR] An unexpected error occurred: {e}")
        # Return a generic server error
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")
    
    finally:
        # 5. This cleanup step ALWAYS runs, deleting the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"  -> Cleaned up temporary file: '{temp_file_path}'")