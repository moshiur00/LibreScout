import streamlit as st
from PIL import Image
from extract_image import extract_image_details
from response_generate import generate_response
from response_generate import search_google
from response_generate import build_faiss_index
from response_generate import query_faiss

def load_interface():
    # Initialize session state for image details
    if "image_details" not in st.session_state:
        st.session_state.image_details = ""

    st.title("Generate Book Information From Image")

    # File uploader for image upload
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        if "uploaded_file" not in st.session_state or uploaded_file != st.session_state.uploaded_file:
            st.session_state.uploaded_file = uploaded_file

            # Clear previous messages
            st.session_state.messages = []

            # Show loading status while processing the image
            with st.status("Processing image...", state="running"):
                image = Image.open(uploaded_file)
                st.session_state.image_details = extract_image_details(image)
                snippets = search_google(st.session_state.image_details, num_results=10)

                

            st.success("Image processed successfully.")
    else:
        st.stop()

    # Generate response
    print(f"🔎 Searching Google for: {st.session_state.image_details}")
    snippets = search_google(st.session_state.image_details)
    
    print("\n⚙️ Building FAISS index...")
    index, snippet_texts = build_faiss_index(snippets)
    
    print("\n📚 Querying FAISS for best matches...")
    context = query_faiss(index, snippet_texts, st.session_state.image_details)
    
    # print("\n🤖 Generating Final Answer...")
    # final_answer = answer_with_context(st.session_state.image_details, context)
    # print(final_answer)

    response = generate_response([st.session_state.image_details], context, "")
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Display assistant message
    with st.chat_message("assistant"):
        st.write(response)
