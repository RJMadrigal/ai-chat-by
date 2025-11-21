from fastapi import FastAPI
from app.routers.chat import router as chat_router

app = FastAPI(
    title="AI Chatbot created by Josue Madrigal",
    description="Chatbot powered by Llama 3, FastAPI and ChromaDB",
    version="1.0.0"
)

app.include_router(chat_router, prefix="/chat", tags=["Chat"])