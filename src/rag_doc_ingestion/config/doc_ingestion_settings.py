from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class doc_ingestion_settings(BaseSettings):

    DOCUMENTS_DIR:str
    VECTOR_STORE_DIR:str
    COLLECTION_NAME:str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"
        