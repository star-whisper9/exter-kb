import config
from service.embedding_engine import XinferenceEmbedding
from service.storage_engine import KnowledgeBaseManager
from service.document_processor import DocumentProcessor

embedding_engine = XinferenceEmbedding(
    config.EMBEDDING_HOST,
    config.EMBEDDING_UID,
    config.EMBEDDING_APIKEY or None
)

document_processor = DocumentProcessor(
    config.DOCUMENT_STORE_PATH or "/app/data/rag-documents",
)

kb_manager = KnowledgeBaseManager(
    config.QDRANT_HOST,
    embedding_engine
)


def get_kb_manager():
    return kb_manager


def get_document_processor():
    return document_processor


def get_embedding_engine():
    return embedding_engine
