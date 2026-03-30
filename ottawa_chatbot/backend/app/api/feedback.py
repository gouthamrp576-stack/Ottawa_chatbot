from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class Feedback(BaseModel):
    message: str
    rating: int | None = None

@router.post("")
def submit_feedback(payload: Feedback):
    # TODO: store in DB or file
    return {"received": True}
