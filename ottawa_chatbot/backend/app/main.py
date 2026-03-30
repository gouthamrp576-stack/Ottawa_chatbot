from fastapi import FastAPI
from backend.app.api.chat import router as chat_router
from backend.app.api.health import router as health_router
from backend.app.api.feedback import router as feedback_router

app = FastAPI(title="Ottawa Newcomer Chatbot API", version="0.1.0")
app.include_router(health_router, prefix="/health", tags=["health"])
app.include_router(chat_router, prefix="/chat", tags=["chat"])
app.include_router(feedback_router, prefix="/feedback", tags=["feedback"])
