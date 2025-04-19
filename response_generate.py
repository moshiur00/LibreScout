from openai import OpenAI
from sentence_transformers import SentenceTransformer
from serpapi import GoogleSearch
from dotenv import load_dotenv
import os
from extract_image import extract_image_details
from PIL import Image
import faiss
import numpy as np


client = OpenAI()
# Load an embedding model (using Sentence Transformers)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load environment variables
load_dotenv()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")


def search_google(query, num_results=10):
    search = GoogleSearch({
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "num": num_results
    })
    results = search.get_dict()
    snippets = []
    
    if "organic_results" in results:
        for res in results["organic_results"]:
            title = res.get("title", "")
            snippet = res.get("snippet", "")
            snippets.append(f"{title}: {snippet}")
    
    return snippets 

# Build FAISS index from text snippets
def build_faiss_index(snippets):
    embeddings = embedding_model.encode(snippets)
    dimension = embeddings.shape[1]
    
    index = faiss.IndexFlatL2(dimension)  # L2 = Euclidean distance
    index.add(np.array(embeddings))
    
    return index, snippets  # Keep snippets for later lookup

# Query FAISS for most relevant context
def query_faiss(index, snippets, question, top_k=5):
    question_embedding = embedding_model.encode([question])
    distances, indices = index.search(np.array(question_embedding), top_k)
    
    relevant_snippets = [snippets[idx] for idx in indices[0]]
    return "\n".join(relevant_snippets)


def build_prompt(extracted_text, context, question=None):
    base_prompt = f"""
        Context: The following text is extracted from a book page via OCR. It may be incomplete or have minor errors.
        Extracted Text: \"\"\" {extracted_text} \"\"\"
    """
    if context:
        base_prompt += f"""The context is given as following"""
        base_prompt += f"""{context}
        """

    if question:
        base_prompt += f"""Question: {question}
    
    Instruction: Answer based only on the provided text. If the answer is not clearly available, state that the information is incomplete. """
    else:
        base_prompt += """Instruction: Summarize the key information from the text with book name, author, publication year and short summary review. Do not assume missing information. """
    
    return base_prompt.strip()


def generate_response(retrieved_texts, query, max_tokens=150):
    
    context = "\n".join(retrieved_texts)
    prompt = build_prompt(context, query)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=1,
        max_tokens=2048,
        top_p=1
    )
    return response.choices[0].message.content

