import os, uuid
from typing import Dict, List, Optional, Any
from llama_index.core import Document
from llama_index.readers.file import PyMuPDFReader, DocxReader
from util.logger import log


class DocumentProcessor:
    """
    文档处理器，将文档文件分段并返回 LlamaIndex 文档对象列表

    todo: 支持更多文件类型
    todo: 支持按分段标注元数据（更细的元数据）
    todo: 支持传入额外的分段方式
    """

    def __init__(self, document_store_path: str):
        self.document_store_path = document_store_path
        os.makedirs(document_store_path, exist_ok=True)
        self.pdf_reader = PyMuPDFReader()
        self.docx_reader = DocxReader()

    def process_document(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        处理文档类文件，将其分段并返回 LlamaIndex 文档对象列表

        Args:
            file_path: 文件路径
            metadata: 以文件为最小单位的元数据信息

        Returns:
            LlamaIndex 文档对象列表
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"未找到需分段的文件: {file_path}")

        file_ext = os.path.splitext(file_path)[1].lower()

        if metadata is None:
            metadata = {}

        # 基础文件元数据
        base_metadata = {
            "file_name": os.path.basename(file_path),
            "file_type": file_ext,
            "file_path": file_path,
            "doc_id": str(uuid.uuid4())
        }

        # 混合元数据
        doc_metadata = {**base_metadata, **metadata}

        log.info(f"处理文件: {file_path}; 元数据: {doc_metadata}")

        # 按类型读取
        if file_ext == ".pdf":
            documents = self.pdf_reader.load_data(file_path)
        elif file_ext in [".docx", ".doc"]:
            documents = self.docx_reader.load_data(file_path)
        elif file_ext in [".txt"]:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                documents = [Document(text=text)]
        else:
            raise ValueError(f"文件类型不支持: {file_ext}")

        # 附加元数据
        for doc in documents:
            doc.metadata.update(doc_metadata)

        # 保存一份原始文件副本到服务器存储目录
        save_path = os.path.join(self.document_store_path, base_metadata["doc_id"] + file_ext)
        with open(file_path, 'rb') as src, open(save_path, 'wb') as dst:
            dst.write(src.read())

        log.info(f"处理了 {len(documents)} 个分段")
        return documents
