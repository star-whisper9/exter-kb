from fastapi import APIRouter

router = APIRouter()

@router.get('/health')
def health():
    return {"status": "running"}

@router.post('/kb/document')
def add_documents():
    pass

@router.delete('/kb/document')
def delete_documents():
    pass

@router.patch('/kb/document')
def update_documents():
    pass