import os, uuid, re
from pathlib import Path
from typing import Dict, List, Optional, Any
from llama_index.core import Document
from llama_index.readers.file import PyMuPDFReader, DocxReader

import config
from util.logger import log
from llama_index.core.node_parser import SentenceSplitter


class DocumentProcessor:
    """
    文档处理器，将文档文件分段并返回 LlamaIndex 文档对象列表

    todo: 支持更多文件类型
    todo: 支持按分段标注元数据（更细的元数据）
    """

    def __init__(self, document_store_path: str):
        self.document_store_path = document_store_path
        os.makedirs(document_store_path, exist_ok=True)
        self.pdf_reader = PyMuPDFReader()
        self.docx_reader = DocxReader()

    def process_document(
            self,
            file_path: str,
            pattern: Optional[str] = None,
            chunk_size: Optional[int] = 1000,
            metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        处理文档类文件，将其分段并返回 LlamaIndex 文档对象列表

        Args:
            file_path: 文件路径
            pattern: 分段标识符，例如"**"或"\n"
            chunk_size: 每个分段的最大字符数，默认1000
            metadata: 以文件为最小单位的元数据信息

        Returns:
            LlamaIndex 文档对象列表
        """
        max_chunk_size = config.CHUNK_MAX
        file_path_p = Path(file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"未找到需分段的文件: {file_path}")

        # 验证chunk_size不超过最大限制
        if chunk_size and chunk_size > max_chunk_size:
            log.warning(f"指定的chunk_size ({chunk_size}) 超过最大限制 ({max_chunk_size})，将使用最大限制")
            chunk_size = max_chunk_size
        elif not chunk_size:
            log.warning("未指定chunk_size，将使用默认值")
            chunk_size = config.CHUNK_DEFAULT_SIZE

        file_ext = os.path.splitext(file_path)[1].lower()
        if metadata is None:
            metadata = {}

        # 基础文件元数据
        base_metadata = {
            "file_type": file_ext[1:],
            "doc_id": str(uuid.uuid4())
        }

        # 混合元数据
        doc_metadata = {**base_metadata, **metadata}
        log.info(f"处理文件: {file_path}; 元数据: {doc_metadata}")

        # 读取文件内容
        if file_ext == ".pdf":
            raw_documents = self.pdf_reader.load_data(file_path_p)
            text_content = "\n\n".join([doc.get_content() for doc in raw_documents])
        elif file_ext in [".docx", ".doc"]:
            raw_documents = self.docx_reader.load_data(file_path_p)
            text_content = "\n\n".join([doc.get_content() for doc in raw_documents])
        elif file_ext in [".txt"]:
            with open(file_path_p, 'r', encoding='utf-8') as f:
                text_content = f.read()
        else:
            raise ValueError(f"文件类型不支持: {file_ext}")

        # 创建句子分割器用于语义分段
        sentence_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=0
        )

        documents = []

        # 如果指定了分段标识符，按标识符分段
        if pattern:
            # 处理特殊字符的转义
            if pattern == r'\n':
                pattern = '\n'
            elif pattern.startswith('\\'):
                try:
                    pattern = bytes(pattern, "utf-8").decode("unicode_escape")
                except:
                    log.warning(f"无法解析转义序列 {pattern}，将按原样使用")

            # 按标识符分段
            chunks = re.split(pattern, text_content)

            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue

                # 检查分段大小是否超过限制
                if len(chunk) > chunk_size:
                    # 使用语义分割器进一步分段
                    sub_docs = sentence_splitter.split_text(chunk)
                    for j, sub_chunk in enumerate(sub_docs):
                        doc_metadata_with_index = {**doc_metadata, "chunk_index": f"{i}.{j}"}
                        documents.append(Document(text=sub_chunk, metadata=doc_metadata_with_index))
                else:
                    # 直接添加分段
                    doc_metadata_with_index = {**doc_metadata, "chunk_index": str(i)}
                    documents.append(Document(text=chunk, metadata=doc_metadata_with_index))
        else:
            # 如果没有指定分段标识符，仅使用语义分段
            chunks = sentence_splitter.split_text(text_content)
            for i, chunk in enumerate(chunks):
                doc_metadata_with_index = {**doc_metadata, "chunk_index": str(i)}
                documents.append(Document(text=chunk, metadata=doc_metadata_with_index))

        # 保存一份原始文件副本到服务器存储目录
        save_path = os.path.join(self.document_store_path, base_metadata["doc_id"] + file_ext)
        with open(file_path, 'rb') as src, open(save_path, 'wb') as dst:
            dst.write(src.read())

        log.info(f"处理了 {len(documents)} 个分段")
        return documents
