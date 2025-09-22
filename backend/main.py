import shutil
import os
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Import from your other backend files
from .text_extractor import extract_text_from_pdf
from .preprocessor import preprocess_report
from .classifytesting import classify_text
from .config import UPLOAD_DIR, LABEL_REPORT, LABEL_PRESCRIPTION, LABEL_IRRELEVANT

app = FastAPI(title="Document Classification API")

# --- CORS Middleware ---
# This is crucial for allowing your friend's frontend to communicate with your backend.
origins = [
    "http://localhost",
    "http://localhost:3000",  # Example for a React frontend
    "http://localhost:8080",  # Example for a Vue frontend
    "null",                   # Allows opening an HTML file directly from the filesystem
    "ButterflyPretty.pythonanywhere.com",
    "https://document-classifier-m9yb.onrender.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- Ensure upload directory exists ---
# This line creates the 'uploaded_docs' folder if it doesn't already exist.
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- API Endpoints ---

@app.get("/health")
def health_check():
    """
    A simple endpoint to check if the server is running correctly.
    """
    print("Health check endpoint was hit successfully.")
    return JSONResponse(content={"status": "ok"})


@app.post("/upload-document/")
async def upload_document(file: UploadFile = File(...)):
    """
    Receives a document, classifies it, and extracts data if it's a report.
    """
    # 1. Save the uploaded file to a temporary location
    temp_file_path = os.path.join(UPLOAD_DIR, f"temp_{file.filename}")
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"[INFO] File saved temporarily to: '{temp_file_path}'")

        # 2. Extract text from the document (PDF)
        print("[INFO] Step 1: Extracting text...")
        extracted_text = extract_text_from_pdf(temp_file_path)
        if not extracted_text:
            print("[WARNING] No text could be extracted from the document.")
            raise HTTPException(status_code=400, detail="Could not extract any text from the document. It might be empty or corrupted.")

        # 3. Classify the extracted text
        print("[INFO] Step 2: Classifying document...")
        prediction = classify_text(extracted_text)
        print(f"  -> Model prediction: '{prediction}'")

        # 4. Handle the prediction result
        if prediction == LABEL_REPORT:
            print("[INFO] Step 3: Document classified as a Report. Extracting structured data...")
            report_data = preprocess_report(extracted_text)
            
            # Convert DataFrame to a list of dictionaries for JSON response
            data_list = report_data.to_dict(orient='records')

            print(f"  -> [ACCEPTED] Report processed. Found {len(data_list)} items.")
            return JSONResponse(
                content={
                    "status": "success",
                    "type": LABEL_REPORT,
                    "data": data_list,
                    "message": "Report processed successfully."
                }
            )
        
        elif prediction == LABEL_PRESCRIPTION:
            print(f"  -> [ACCEPTED] Document classified as a Prescription.")
            return JSONResponse(
                content={
                    "status": "success",
                    "type": LABEL_PRESCRIPTION,
                    "message": "Prescription submitted successfully."
                }
            )

        else: # The document is irrelevant
            print(f"  -> [REJECTED] Document classified as Irrelevant.")
            raise HTTPException(
                status_code=400,
                detail="The uploaded document is not a valid report or prescription."
            )

    except Exception as e:
        print(f"[CRITICAL ERROR] An unexpected error occurred: {e}")
        # Print the full traceback to the console for detailed debugging
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

    finally:
        # 5. This cleanup step ALWAYS runs, deleting the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"  -> Cleaned up temporary file: '{temp_file_path}'")