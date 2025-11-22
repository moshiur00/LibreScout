import streamlit as st
from PIL import Image
import json
import re
import ast

from extract_image import extract_image_details
from response_generate import (
    generate_response as generate_book_info_response,
    search_google,
    build_faiss_index,
    query_faiss,
    search_book_offers,
)
from llm_response import generate_response_with_search
from llm_response import generate_response_40

from prompt_builder import build_book_search_prompt


def extract_json_like(text: str) -> str:
    """
    Extract a JSON / Python-literal array or object from a model response.

    - Removes ```json ... ``` or ``` ... ``` code fences if present
    - Otherwise returns the raw text trimmed
    """
    if not isinstance(text, str):
        return text

    # Look for fenced code blocks ```json ...``` or ``` ...```
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return text.strip()


def parse_books_response(raw_text: str):
    """
    Try to parse the model output into a Python list of book dicts.

    1. Strip markdown fences.
    2. Try strict JSON via json.loads.
    3. If that fails, try ast.literal_eval for Python-style 'JSON'.
    """
    cleaned = extract_json_like(raw_text)

    # 1) Try strict JSON
    try:
        data = json.loads(cleaned)
    except Exception:
        # 2) Try Python literal parsing (handles single quotes, etc.)
        try:
            data = ast.literal_eval(cleaned)
        except Exception:
            return None  # caller will handle error

    # Normalize to list
    if isinstance(data, dict):
        data = [data]
    return data


def load_interface():
    # Initialize session state
    if "image_details" not in st.session_state:
        st.session_state.image_details = ""
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.title("📚 Book Finder & Analyzer")

    # --------------------------------------------------------
    # 1️⃣ FIRST SECTION — USER CAN TYPE BOOK TASTE
    # --------------------------------------------------------
    st.subheader("Find a book that matches your taste first")

    user_preference = st.text_input(
        "Describe what kind of book you want "
        "(e.g. 'dark fantasy with political intrigue', 'light romance, funny', 'short sci-fi story')."
    )

    if user_preference:
        with st.status("Searching for books that match your taste...", state="running"):
            search_prompt = build_book_search_prompt(user_preference)
            print(search_prompt)
            raw_json = generate_response_40(search_prompt)

        books = parse_books_response(raw_json)

        if books is None:
            st.error("Couldn't parse book list. Raw model output:")
            st.code(raw_json)
            books = []

        st.markdown("---")
        st.subheader("Recommended Books for Your Taste")

        if books:
            for book in books:
                title = book.get("title", "Unknown")
                author = book.get("author", "Unknown")
                description = book.get("description")
                language = book.get("language", "Unknown")
                genres = book.get("genres", [])
                relevance = book.get("relevance", "")

                with st.container():
                    st.markdown(f"### {title}")
                    st.write(f"**Author:** {author}")
                    st.write(f"**Language:** {language}")
                    if genres:
                        st.write(f"**Genres:** {', '.join(genres)}")
                    if description:
                        st.write(f"**Description:** {description}")
                    if relevance:
                        st.write(f"**Reason:** {relevance}")
        else:
            st.info("No matching books found.")

    # --------------------------------------------------------
    # 2️⃣ SECOND SECTION — IMAGE UPLOAD
    # --------------------------------------------------------
    st.markdown("---")
    st.subheader("Or upload an image of a book")

    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        return

    if uploaded_file != st.session_state.uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        with st.status("Processing image...", state="running"):
            image = Image.open(uploaded_file)
            st.session_state.image_details = extract_image_details(image)
        st.success("Image processed successfully.")

    # Display OCR results
    image = Image.open(st.session_state.uploaded_file)
    col_img, col_text = st.columns([1, 2])
    with col_img:
        st.image(image, caption="Uploaded image", use_container_width=True)
    with col_text:
        st.markdown("**Extracted Text (OCR):**")
        st.write(st.session_state.image_details)

    # --------------------------------------------------------
    # 3️⃣ BOOK INFORMATION FROM IMAGE
    # --------------------------------------------------------
    st.markdown("---")
    st.subheader("Book Information From Image")

    snippets = search_google(st.session_state.image_details)
    index, snippet_texts = build_faiss_index(snippets)
    context = query_faiss(index, snippet_texts, st.session_state.image_details)

    response = generate_book_info_response(
        [st.session_state.image_details], context, ""
    )
    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.write(response)

    # --------------------------------------------------------
    # 4️⃣ WHERE TO BUY THE BOOK
    # --------------------------------------------------------
    st.markdown("---")
    st.subheader("Where to buy this book")

    offers = search_book_offers(st.session_state.image_details, num_results=5)

    if not offers:
        st.info("No purchase options found.")
        return

    for offer in offers:
        title = offer.get("title", "Offer")
        price = offer.get("price")
        rating = offer.get("rating")
        reviews = offer.get("reviews")
        source = offer.get("source")
        link = offer.get("link")

        with st.container():
            st.markdown("---")
            cols = st.columns([3, 1])

            with cols[0]:
                st.markdown(f"**{title}**")
                meta = []
                if price:
                    meta.append(f"💰 {price}")
                if rating:
                    if reviews:
                        meta.append(f"⭐ {rating}/5 ({reviews} reviews)")
                    else:
                        meta.append(f"⭐ {rating}/5")
                if source:
                    meta.append(f"📦 {source}")
                st.markdown(" • ".join(meta))

            with cols[1]:
                if link:
                    st.link_button("View offer", link, use_container_width=True)
