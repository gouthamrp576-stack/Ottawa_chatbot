from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseModel):
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    openai_vector_store_id: str = os.getenv("OPENAI_VECTOR_STORE_ID", "")
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "ottawa_newcomer")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

settings = Settings()
