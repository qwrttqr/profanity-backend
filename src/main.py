from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.main import api_router
from src.utils import TextAnalyzer
from db.utils.init_db import connect_db
from src.utils.post_learn.profanity_module import ProfanityModule
from src.utils.post_learn.semantic_module import SemanticModule

app = FastAPI()
@app.on_event("startup")
async def startup():
    profanity_module = ProfanityModule()
    semantic_module = SemanticModule()
    app.state.profanity_module = profanity_module
    app.state.semantic_module = semantic_module
    app.state.analyzer = TextAnalyzer(profanity_module, semantic_module)


try:
    connect_db()
except Exception as e:
    print('Error connecting to db', e)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
