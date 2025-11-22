from openai import OpenAI
from sentence_transformers import SentenceTransformer
from serpapi import GoogleSearch
from dotenv import load_dotenv
import os
from extract_image import extract_image_details
from PIL import Image
import faiss
import numpy as np


load_dotenv()
client = OpenAI()
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
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


def search_book_offers(book_query, num_results=5):
    """
    Use SerpAPI to fetch Google Shopping results for the book.
    Returns a list of offer dicts with title, price, rating, reviews, link, source.
    """
    if not SERPAPI_API_KEY:
        return []

    search = GoogleSearch({
        "q": book_query,
        "tbm": "shop",
        "num": num_results,
        "api_key": SERPAPI_API_KEY,
    })

    try:
        results = search.get_dict()
    except Exception:
        return []

    offers = []
    for item in results.get("shopping_results", []):
        offers.append({
            "title": item.get("title"),
            "price": item.get("price") or item.get("extracted_price"),
            "rating": item.get("rating"),
            "reviews": item.get("reviews"),
            "link": item.get("product_link") or item.get("link"),
            "source": item.get("source"),
        })

    return offers


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
    """
    extracted_text: main text about the book (e.g., BLIP caption or OCR)
    context: additional web snippets (from Google / RAG)
    """
    base_prompt = f"""
        Context: The following text is extracted from or related to a book (e.g., cover, description, quotes). 
        It may be incomplete or have minor errors.
        Extracted Text: \"\"\" {extracted_text} \"\"\"
    """
    if context:
        base_prompt += f"""
        Additional web snippets about the same book:
        {context}
        """

    if question:
        base_prompt += f"""
        Question: {question}

        Instruction: Answer based only on the provided text and snippets above.
        If the answer is not clearly available, explicitly say that the information is incomplete.
        Do NOT invent details.
        """
    else:
        base_prompt += """
        Instruction: You are a book expert. Based only on the text above:
        - Identify the most likely book title and author (or say "Unknown" if unclear).
        - Provide, if available:
          • Publication year
          • Main genre
          • 3–5 bullet points summarizing what the book is about
          • A short, neutral note on who this book is for (target audience).
        If any item cannot be determined from the text, say that it is not clearly specified.
        Do NOT assume or invent missing information.
        """

    return base_prompt.strip()


def generate_response(retrieved_texts, query, max_tokens=150):
    # retrieved_texts is usually [image_description]; query is usually the RAG context
    context = "\n".join(retrieved_texts)
    prompt = build_prompt(context, query)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant and book expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=1,
        max_tokens=2048,
        top_p=1
    )
    return response.choices[0].message.content
