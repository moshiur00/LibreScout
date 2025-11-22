import streamlit as st
from PIL import Image

from extract_image import extract_image_details
from response_generate import (
    generate_response,
    search_google,
    build_faiss_index,
    query_faiss,
    search_book_offers,  # NEW
)


def load_interface():
    # Initialize session state
    if "image_details" not in st.session_state:
        st.session_state.image_details = ""
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.title("Generate Book Information From Image")

    # --- Image upload ---
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        st.stop()

    # If a new file is uploaded, process it
    if uploaded_file != st.session_state.uploaded_file:
        st.session_state.uploaded_file = uploaded_file

        with st.status("Processing image...", state="running"):
            image = Image.open(uploaded_file)
            st.session_state.image_details = extract_image_details(image)

        st.success("Image processed successfully.")

    # Show the image and extracted details
    image = Image.open(st.session_state.uploaded_file)
    col_img, col_text = st.columns([1, 2])
    with col_img:
        st.image(image, caption="Uploaded image", use_container_width=True)
    with col_text:
        st.markdown("**Extracted details from image (OCR):**")
        st.write(st.session_state.image_details)

    # --- Search & Book summary ---
    st.markdown("---")
    st.subheader("Book information")

    snippets = search_google(st.session_state.image_details)
    index, snippet_texts = build_faiss_index(snippets)
    context = query_faiss(index, snippet_texts, st.session_state.image_details)

    # Generate LLM response with book info
    response = generate_response([st.session_state.image_details], context, "")
    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        # st.write renders basic Markdown nicely
        st.write(response)

    # --- Where to buy section ---
    st.markdown("---")
    st.subheader("Where to buy this book")

    # Use the image description / OCR text as a query to Google Shopping
    offers = search_book_offers(st.session_state.image_details, num_results=5)

    if not offers:
        st.info("No purchase options found for this book.")
        return

    for offer in offers:
        title = offer.get("title", "Offer")
        price = offer.get("price")
        rating = offer.get("rating")
        reviews = offer.get("reviews")
        source = offer.get("source")
        link = offer.get("link")

        with st.container():
            # A small separator between offers
            st.markdown("---")

            cols = st.columns([3, 1])

            # Left: title + meta info
            with cols[0]:
                st.markdown(f"**{title}**")

                meta_bits = []
                if price:
                    meta_bits.append(f"💰 {price}")
                if rating:
                    if reviews:
                        meta_bits.append(f"⭐ {rating}/5 ({reviews} reviews)")
                    else:
                        meta_bits.append(f"⭐ {rating}/5")
                if source:
                    meta_bits.append(f"📦 {source}")

                if meta_bits:
                    st.markdown(" • ".join(meta_bits))

            # Right: button only, no raw URL
            with cols[1]:
                if link:
                    # This shows a clean button instead of the full link
                    st.link_button("View offer", link, use_container_width=True)
