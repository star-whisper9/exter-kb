from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.node_parser import SentenceSplitter
from qdrant_client import QdrantClient, models
from typing import List, Dict, Any, Optional
from llama_index.core import Document
from util.logger import log
import os
import json

import config


def convert_metadata_dict_to_filters(metadata_filter: Dict[str, Any]) -> MetadataFilters:
    """
    将元数据字典转换为 LlamaIndex MetadataFilters
    """
    filters = [
        MetadataFilter(
            key=key,
            value=value
        )
        for key, value in metadata_filter.items()
    ]
    return MetadataFilters(filters=filters)


class KnowledgeBase:
    """
    使用 Qdrant 作为存储引擎的知识库
    """

    def __init__(
            self,
            qdrant_host: str,
            collection_name: str,
            embedding_engine,
            chunk_size: int = 1000,
            chunk_overlap: int = 100,
            persist_dir: str = config.INDEX_STORE_PATH
    ):
        self.qdrant_host = qdrant_host
        self.collection_name = collection_name
        self.embedding_engine = embedding_engine
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_dir = persist_dir

        # 初始化 Qdrant
        self.client = QdrantClient(url=qdrant_host)
        self._initialize_collection()

        # 初始化 Qdrant 向量存储
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            metadata_payload_key="metadata"
        )

        # 初始化节点解析器
        self.node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # 检查是否有现有索引
        # 注意，如果索引目录已存在，但目录中没有正确索引，会报错
        if os.path.exists(persist_dir):
            log.info(f"读取已有索引： {persist_dir}")
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store,
                persist_dir=persist_dir
            )
            self.index = load_index_from_storage(
                storage_context,
                embed_model=self.embedding_engine.get_embedding_model()  # 显式指定嵌入模型
            )
        else:
            log.info("创建新索引中")
            os.makedirs(persist_dir, exist_ok=True)
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                embed_model=self.embedding_engine.get_embedding_model(),
            )
            self.index.storage_context.persist(persist_dir=persist_dir)

    def _initialize_collection(self):
        """检查并初始化Qdrant集合"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]

            # 获取嵌入模型的向量维度
            vector_size = self.embedding_engine.get_embedding_dimension()

            # 如果集合不存在，创建集合
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                log.info(f"创建了 '{self.collection_name}' 集合。向量维度: {vector_size}")
            else:
                log.info(f"集合 '{self.collection_name}' 已存在")
        except Exception as e:
            log.error(f"初始化向量集合出错: {e}")
            raise

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        新增文档到知识库

        Args:
            documents: LlamaIndex 文档列表

        Returns:
            存储的文档 ID 列表
        """
        log.info(f"正在添加 {len(documents)} 个文段到知识库")

        # Parse nodes (chunking)
        nodes = self.node_parser.get_nodes_from_documents(documents)

        # Add to index
        for node in nodes:
            if not hasattr(node, 'metadata'):
                node.metadata = {}

        # Insert into index
        self.index.insert_nodes(nodes)

        # Persist to disk
        self.index.storage_context.persist(persist_dir=self.persist_dir)

        # Return document IDs
        return [doc.metadata.get("doc_id") for doc in documents]

    def vector_search(self, query: str, top_k: int = 5, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        向量相似度检索

        Args:
            query: 检索词
            top_k: 返回的最高分数的结果数量
            metadata_filter: (可选) 元数据过滤器

        Returns:
            检索结果列表(score, metadata, text)
        """
        log.info(f"进行向量检索: {query}; 过滤器: {metadata_filter}")

        filters = None
        if metadata_filter:
            filters = convert_metadata_dict_to_filters(metadata_filter)
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k,
            filters=filters
        )

        nodes = retriever.retrieve(query)

        return [{
            "score": node.score if hasattr(node, 'score') else None,
            "metadata": node.metadata,
            "text": node.text,
        } for node in nodes]

    def metadata_search(self, metadata_filter: Dict[str, Any], limit: int = 10) -> List[Dict]:
        """
        Qdrant 元数据检索
        Args:
            metadata_filter: 元数据过滤器
        Returns:
            检索结果列表(id, metadata, text)
        """
        log.info(f"使用元数据过滤检索: {metadata_filter}")

        must_conditions = [
            models.FieldCondition(
                key=key,
                match=models.MatchValue(value=value)
            )
            for key, value in metadata_filter.items()
        ]

        records, point_id = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(must=must_conditions),
            limit=limit
        )

        return [{
            "id": point.id,
            "metadata": json.loads(point.payload.get("_node_content", "")).get("metadata", ""),
            "text": json.loads(point.payload.get("_node_content", "")).get("text", "")
        } for point in records]
