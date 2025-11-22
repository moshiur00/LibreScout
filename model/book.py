from pydantic import BaseModel

class Book(BaseModel):
    title: str
    author: str
    description: str | None = None
    language: str
    genres: list[str] = []
    relevance: str