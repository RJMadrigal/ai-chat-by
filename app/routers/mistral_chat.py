from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from app.models.message import Message, ChatResponse
from app.services.mistral_service import MistralService

router = APIRouter()
mistral_service_instance = None

def get_mistral_service():
    global mistral_service_instance
    if mistral_service_instance is None:
        mistral_service_instance = MistralService()
    return mistral_service_instance

@router.post("/", response_model=ChatResponse)
async def chat_with_mistral(
    payload: Message,
    mistral_service: MistralService = Depends(get_mistral_service)
):
    """Chat con Mistral AI (respuesta completa)"""
    response = await mistral_service.generate_answer(
        user_id=payload.user_id,
        query=payload.message
    )
    return ChatResponse(answer=response)

@router.post("/stream")
async def chat_with_mistral_stream(
    payload: Message,
    mistral_service: MistralService = Depends(get_mistral_service)
):
    """Chat con Mistral AI (streaming)"""
    return StreamingResponse(
        mistral_service.generate_answer_stream(
            user_id=payload.user_id,
            query=payload.message
        ),
        media_type="text/plain"
    )