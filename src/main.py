from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.main import api_router
from src.utils import TextAnalyzer
from db.utils.init_db import connect_db
from src.utils.profanity_module import ProfanityModule
from src.utils.load import model_path, vectorizer_path

app = FastAPI()
@app.on_event("startup")
async def startup():
    profanity_module = ProfanityModule(model_path, vectorizer_path)
    app.state.profanity_module = profanity_module
    app.state.analyzer = TextAnalyzer(profanity_module)


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
