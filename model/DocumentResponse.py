from typing import List

from pydantic import BaseModel


class DocumentResponse(BaseModel):
    status: str
    document_ids: List[str]
    failed_files: List[str] = []
