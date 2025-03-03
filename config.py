import os

# App Config
APP_HOST = ""
APP_PORT = 8080
LOGGING_LEVEL = "DEBUG"
LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Qdrant 配置
QDRANT_HOST = "http://192.168.46.141:6333"
COLLECTION_NAME = "test_collection"

# Xinference 配置
EMBEDDING_HOST = "http://192.168.46.141:9997"
EMBEDDING_UID = "bge-m3"
EMBEDDING_APIKEY = ""
EMBEDDING_MODEL_NAME = "bge-m3"

# 服务器持久化路径
DOCUMENT_STORE_PATH = os.path.join(os.path.dirname(__file__), "data/documents")
INDEX_STORE_PATH = os.path.join(os.path.dirname(__file__), "data/indexes")

# 分段配置
CHUNK_DEFAULT_SIZE = 1000
CHUNK_DEFAULT_OVERLAP = 100
CHUNK_MAX = 10000
BATCH_SIZE = 10
