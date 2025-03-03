from fastapi import APIRouter
from api.endpoints import kb_management, query

router = APIRouter()

router.include_router(kb_management.router, prefix="/api", tags=["Knowledge Base Management"])
router.include_router(query.router, prefix="/api", tags=["Query"])
