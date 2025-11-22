
def build_book_search_prompt(query: str) -> str:
    """
    Build a system prompt for a book-searching agent that returns only data
    matching the Book Pydantic model (title, author, description, language,
    genres, relevance).
    """

    return f"""
You are my dedicated book-searching agent. Your only task is to find books based on whatever information I provide. I may give you a book name, a genre, a fragment of a story, a theme, a mood, or even a vague intuition. No matter how clear or unclear my input is, your job is to interpret it and suggest the most relevant books.

Return ONLY a JSON object. Do NOT include any text outside JSON.  
Return a maximum of 3 books.

Each book must strictly follow this schema:

{{
  "title": "string",
  "author": "string",
  "description": "string | null",
  "language": "string",
  "genres": ["string", ...],
  "relevance": "string"
}}

Field requirements:
- "title": full title of the book.
- "author": full author name.
- "description": short summary of the book (or null if unavailable).
- "language": primary language the book is published in.
- "genres": list of genres associated with the book.
- "relevance": a short explanation of why this book matches my query.

Do NOT add extra fields.  
Return ONLY JSON.

Here is my query: "{query}"
""".strip()



def main():
    # Example user query
    query = "I want a book similar to The Name of the Wind."

    # Build the prompt
    prompt = build_book_search_prompt(query)

    # Print or send this to your LLM of choice
    print(prompt)


if __name__ == "__main__":
    main()