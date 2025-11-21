from fastapi import APIRouter, Depends
from app.models.message import Message, ChatResponse
from app.services.chatbot_service import ChatbotService

router = APIRouter()
chat_service_instance = ChatbotService()

def get_chat_service():
    return chat_service_instance

@router.post("/", response_model=ChatResponse)
async def chat_with_bot(
    payload: Message,
    chat_service: ChatbotService = Depends(get_chat_service)
):
    response = await chat_service.generate_answer(
        user_id=payload.user_id,
        query=payload.message
    )
    return ChatResponse(answer=response)
