from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
import os
import json
import re
import ast
import numpy as np


from extract_image import extract_image_details
from response_generate import (
    generate_response as generate_book_info_response,
    search_google,
    build_faiss_index,
    query_faiss,
    search_book_offers,
)
from llm_response import generate_response_40
from prompt_builder import build_book_search_prompt

app = Flask(__name__)

# Where uploaded images will be stored for preview
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ---------- Helpers for parsing the LLM JSON output ---------- #

def extract_json_like(text: str) -> str:
    """
    Extract a JSON / Python-literal array or object from a model response.

    - Removes ```json ... ``` or ``` ... ``` code fences if present
    - Otherwise returns the raw text trimmed
    """
    if not isinstance(text, str):
        return text

    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return text.strip()


def parse_books_response(raw_text: str):
    """
    Try to parse the model output into a Python list of book dicts.
    """
    cleaned = extract_json_like(raw_text)

    try:
        data = json.loads(cleaned)
    except Exception:
        try:
            data = ast.literal_eval(cleaned)
        except Exception:
            return None

    if isinstance(data, dict):
        data = [data]
    return data


# ------------------------ Routes ------------------------ #

@app.route("/", methods=["GET", "POST"])
def index():
    # shared state for every request
    user_preference = ""
    recommended_books = []
    books_error_raw = None

    image_url = None
    ocr_text = None
    ocr_np = None
    book_info = None
    offers = []

    # mode + validation messages
    search_mode = "preference"
    preference_error = None
    image_error = None

    if request.method == "POST":
        # read which radio button was selected
        search_mode = request.form.get("search_mode") or "preference"

        # ----- MODE 1: Search by preference -----
        if search_mode == "preference":
            user_preference = (request.form.get("user_preference") or "").strip()

            if not user_preference:
                preference_error = "Please describe what kind of book you want."
            else:
                try:
                    search_prompt = build_book_search_prompt(user_preference)
                    raw_json = generate_response_40(search_prompt)
                    books = parse_books_response(raw_json)

                    if books is None:
                        books_error_raw = raw_json
                    else:
                        recommended_books = books
                except Exception as e:
                    books_error_raw = f"Error while requesting recommendations: {e}"

        # ----- MODE 2: Analyze book cover -----
        elif search_mode == "image":
            uploaded_file = request.files.get("book_image")

            if not uploaded_file or not uploaded_file.filename:
                image_error = "Please upload a book cover image."
            else:
                filename = secure_filename(uploaded_file.filename)
                save_path = os.path.join(UPLOAD_FOLDER, filename)

                # Save image for preview
                uploaded_file.stream.seek(0)
                image = Image.open(uploaded_file.stream).convert("RGB")
                image.save(save_path)
                image_url = "/" + save_path.replace("\\", "/")

                # OCR + downstream calls
                try:
                    ocr_np = np.array(image)
                    ocr_text = extract_image_details(ocr_np)
                    # ocr_text = extract_image_details(image)

                    snippets = search_google(ocr_text)
                    index, snippet_texts = build_faiss_index(snippets)
                    context = query_faiss(index, snippet_texts, ocr_text)

                    book_info = generate_book_info_response([ocr_text], context, "")
                    offers = search_book_offers(ocr_text, num_results=5)
                except Exception as e:
                    book_info = f"An error occurred while analyzing the image: {e}"

        # (no else: only two modes right now)

    return render_template(
        "index.html",
        search_mode=search_mode,
        user_preference=user_preference,
        recommended_books=recommended_books,
        books_error_raw=books_error_raw,
        image_url=image_url,
        ocr_text=ocr_text,
        book_info=book_info,
        offers=offers,
        preference_error=preference_error,
        image_error=image_error,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
