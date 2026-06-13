import logging
import chromadb
from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,StorageContext
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.rag_doc_ingestion.config.doc_ingestion_settings import doc_ingestion_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

settings = doc_ingestion_settings
logger.info("Loading HuggingFace Embedding model")
embed_model = HuggingFaceEmbedding()

def build_vector_store_from_documents():
    logger.info("Starting vector store ingestion process")
    try:
        docs_dir_path=settings.DOCUMENTS_DIR
        vector_store_path=settings.VECTOR_STORE_DIR
        collection_name=settings.COLLECTION_NAME
        logger.info(f"Loading Documents directory path: {docs_dir_path}")
        loader=SimpleDirectoryReader(input_dir=docs_dir_path,)
        documents=loader.load_data()

        parser=SimpleNodeParser().from_defaults(chunk_size=1024,chunk_overlap=50)
        logger.info("parsing documents into nodes")
        nodes=parser.get_nodes_from_documents(documents)
        logger.info(f"parsed {len(nodes)} nodes.")
        logger.info(f"initializing ChromaDB Persistent client at:{vector_store_path}")
        db=chromadb.persistentClient(path=vector_store_path)
        chroma_collection = db.get_or_collection(name=collection_name)
        logger.info(f"Creating chroma vector store with collection path: {collection_name}")
        vector_store=ChromaVectorStore(collection_name=collection_name)

        storage_context=StorageContext.from_defaults(vector_store=vector_store)
        logger.info("building vector store index.")
        index=VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            vector_store=vector_store,
            embedding=embed_model,
        )
        logger.info(f"Vector store build completed successfully")
        return 0
    except Exception as e:
        logger.error(f"error during vector store build: {e}")
        return 1
if __name__ == "__main__":
    build_vector_store_from_documents()
