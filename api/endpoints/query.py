from fastapi import APIRouter

router = APIRouter()

@router.post('/exact_retrieval')
def exact_retrival():
    pass

@router.post('/retrieval')
def vector_retrival():
    pass