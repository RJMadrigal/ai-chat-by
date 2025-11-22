from fastapi import FastAPI
from dotenv import load_dotenv
from app.routers.chat import router as chat_router
from app.routers.mistral_chat import router as mistral_router

app = FastAPI(
    title="AI Chatbot created by Josue Madrigal",
    description="Chatbot powered by Llama 3 & MistralAI, FastAPI and ChromaDB",
    version="1.0.0"
)

app.include_router(chat_router, prefix="/chat", tags=["Chat"])
app.include_router(mistral_router, prefix="/mistral", tags=["Chat - MistralAI"])   


@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "online",
        "endpoints": {
            "ollama": "/chat",
            "mistral": "/mistral",
            "mistral_stream": "/mistral/stream"
        }
    }