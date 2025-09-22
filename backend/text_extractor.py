# import pytesseract
# from PyPDF2 import PdfReader
# from pdf2image import convert_from_path

# # Path to Tesseract (change if needed)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# def extract_text_from_pdf(pdf_path: str, lang="eng") -> str:
#     text = ""
#     try:
#         reader = PdfReader(pdf_path)
#         for page in reader.pages:
#             page_text = page.extract_text()
#             if page_text:
#                 text += page_text + "\n"
#     except Exception as e:
#         print(f"[Warning] PyPDF2 failed: {e}")

#     if not text.strip():
#         print("[Info] No digital text found. Using OCR...")
#         pages = convert_from_path(pdf_path)
#         for i, page in enumerate(pages, start=1):
#             print(f"[OCR] Processing page {i}...")
#             ocr_text = pytesseract.image_to_string(page, lang=lang)
#             text += ocr_text + "\n"

#     return text.strip()

#-------------------------------------------------------------------------------------------------------------
#for linux - render

import pytesseract
from PyPDF2 import PdfReader
from pdf2image import convert_from_path

# On Render's Linux servers, Tesseract will be installed in the system's PATH.
# We must REMOVE the hardcoded Windows path.
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" # <-- DELETE THIS LINE

def extract_text_from_pdf(pdf_path: str, lang="eng") -> str:
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        print(f"[Warning] PyPDF2 failed: {e}")

    if not text.strip():
        print("[Info] No digital text found. Using OCR...")
        try:
            pages = convert_from_path(pdf_path)
            for i, page in enumerate(pages, start=1):
                ocr_text = pytesseract.image_to_string(page, lang=lang)
                text += ocr_text + "\n"
        except Exception as ocr_error:
            print(f"[ERROR] OCR processing failed: {ocr_error}")
            return text.strip()

    return text.strip()
