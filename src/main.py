from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.main import api_router
from src.utils import TextAnalyzer
from db.utils.init_db import connect_db
from src.utils.post_learn.profanity_module import ProfanityModule
from src.utils.post_learn.semantic_module import SemanticModule
from src.utils.file_work import FileManager

app = FastAPI()
@app.on_event("startup")
async def startup():
    FileManager() # Initializing FileManager class initializes all files too
    profanity_module = ProfanityModule()
    semantic_module = SemanticModule()
    app.state.profanity_module = profanity_module
    app.state.semantic_module = semantic_module
    app.state.analyzer = TextAnalyzer(profanity_module, semantic_module)


try:
    connect_db()
except Exception as e:
    print(f'Error connecting to db {str(e)}')


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
