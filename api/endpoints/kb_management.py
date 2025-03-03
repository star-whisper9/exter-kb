import json
import os
import tempfile
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, File, UploadFile, Form, Depends, HTTPException

from dependencies import get_kb_manager, get_document_processor
from model.DocumentResponse import DocumentResponse
from service.document_processor import DocumentProcessor
from service.storage_engine import KnowledgeBaseManager

from util.logger import log

router = APIRouter()


@router.get('/health')
def health():
    return {"status": "running"}


@router.post('/kb/document', response_model=DocumentResponse)
async def add_documents(
        background_tasks: BackgroundTasks,
        files: List[UploadFile] = File(...),
        kb_id: str = Form(...),
        metadata: Optional[str] = Form(None),
        custom_delimiter: Optional[str] = Form(None),
        chunk_size: Optional[int] = Form(1000),
        kb_manager: KnowledgeBaseManager = Depends(get_kb_manager),
        doc_processor: DocumentProcessor = Depends(get_document_processor)
):
    """
    向知识库增加文档

    Args:
        files: 上传的文件列表
        kb_id: 知识库ID
        metadata: (可选) 元数据(应用于所有文件)
        custom_delimiter: (可选) 自定义文本分段符
        chunk_size: (可选) 分块大小

        kb_manager: - 知识库管理器
        doc_processor: - 文档处理器
        background_tasks: - 后台任务

    Returns:
        200: 成功
        500: 失败
        400: 参数错误

    """
    try:
        # 解析元数据
        metadata_dict = {}
        if metadata:
            try:
                metadata_dict = json.loads(metadata)
                if not isinstance(metadata_dict, dict):
                    raise HTTPException(status_code=400, detail="元数据必须是有效的JSON对象")
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="无法解析元数据JSON")

        # 获取或创建知识库
        kb = kb_manager.get_knowledge_base(kb_id)
        if not kb:
            kb = kb_manager.create_knowledge_base(kb_id)
            log.info(f"创建新知识库: {kb_id}")

        # 保存上传的文件并处理
        document_ids = []
        failed_files = []

        for file in files:
            try:
                # 创建临时文件
                suffix = os.path.splitext(file.filename)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    temp_path = temp_file.name
                    # 写入上传的内容
                    content = await file.read()
                    temp_file.write(content)

                # 添加文件名到元数据
                file_metadata = {
                    **metadata_dict,
                    "original_filename": file.filename
                }

                # 处理文档
                documents = doc_processor.process_document(
                    file_path=temp_path,
                    pattern=custom_delimiter,
                    chunk_size=chunk_size,
                    metadata=file_metadata
                )

                # 添加到知识库
                doc_ids = kb.add_documents(documents)
                document_ids.extend(doc_ids)

                # 清理临时文件
                background_tasks.add_task(os.remove, temp_path)

            except Exception as e:
                log.error(f"处理文件 {file.filename} 时出错: {str(e)}")
                failed_files.append(file.filename)
                # 继续处理其他文件

        if not document_ids and failed_files:
            # 所有文件处理失败
            raise HTTPException(status_code=500, detail="所有文件处理失败")

        return DocumentResponse(
            status="success" if document_ids else "partial_success",
            document_ids=document_ids,
            failed_files=failed_files
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"添加文档时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")


@router.delete('/kb/document')
def delete_documents():
    """
    删除知识库文档

    Args:

    Returns:

    """

    pass


@router.patch('/kb/document')
def update_documents():
    """
    更新知识库文档

    Args:

    Returns:

    """
    pass
