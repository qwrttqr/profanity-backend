from fastapi import APIRouter
from src.api.routes import analyze

api_router = APIRouter()
api_router.include_router(analyze.router)
