from __future__ import annotations

from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter

from backend.app.rag.generator import generate_answer

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    category: Optional[str] = None


@router.post("/chat")
async def chat(req: ChatRequest):
    return await generate_answer(req.message, category=req.category)