from langchain_community.embeddings import XinferenceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from util.logger import log
from typing import Optional

import config


class XinferenceEmbedding:
    """
    通过 LangChain Adapter 封装 Xinference 嵌入模型

    todo: 异步嵌入实现
    """

    def __init__(
            self,
            server_url: str,
            model_uid: str,
            api_key: Optional[str] = None
    ):
        """
        初始化 Xinference 嵌入模型

        Args:
            server_url: Xinference 端点
            model_uid: 模型 UID
            api_key: (可选) Xinference API 密钥
        """
        try:
            log.info(f"初始化 Xinference 嵌入模型 {model_uid} 中")
            xinference_embed = XinferenceEmbeddings(
                server_url=server_url,
                model_uid=model_uid
            )

            if api_key:
                xinference_embed.client.headers.update({"Authorization": f"Bearer {api_key}"})

            # 包装为 LlamaIndex 支持的 LangChain 嵌入模型
            self.embedding_model = LangchainEmbedding(
                langchain_embeddings=xinference_embed,
                embed_batch_size=config.BATCH_SIZE
            )

            log.info("嵌入模型初始化成功")

        except Exception as e:
            log.error(f"嵌入模型初始化失败: {str(e)}")
            raise

    def get_embedding_model(self):
        """返回嵌入模型"""
        return self.embedding_model

    def get_embedding_dimension(self) -> int:
        """
        获取嵌入向量的维度

        Returns:
            int: 嵌入向量的维度
        """
        # 根据使用的模型返回正确的维度
        model_dimensions = {
            "bge-large-zh": 1024,
            "bge-base-zh": 768,
            "text-embedding-ada-002": 1536,
            # 添加更多模型及其维度
        }

        # 如果模型在预设列表中，返回对应维度
        model_name = config.XEMBEDDING_MODEL_NAME
        if model_name in model_dimensions:
            return model_dimensions[model_name]

        # 如果不在预设列表中，通过嵌入测试文本来获取维度
        test_embedding = self.get_text_embedding("测试文本")
        log.info("获取嵌入维度失败，通过测试文本获取维度")
        return len(test_embedding)

    def get_text_embedding(self, text: str) -> Optional[list]:
        """
        获取文本的嵌入向量

        Args:
            text: 待嵌入的文本

        Returns:
            list: 文本的嵌入向量
        """
        return self.embedding_model.get_text_embedding(text)

    def get_text_embeddings(self, texts: list) -> Optional[list]:
        """
        获取文本列表的嵌入向量

        Args:
            texts: 待嵌入的文本列表

        Returns:
            list: 文本列表的嵌入向量
        """
        return self.embedding_model.get_text_embeddings(texts)
