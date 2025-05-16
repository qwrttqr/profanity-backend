from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.main import api_router
from db.init_db import connect_db

app = FastAPI()
# try:
#     connect_db()
# except Exception as e:
#     print('Error connecting to db', e)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)