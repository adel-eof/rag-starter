from pydantic import BaseModel
from typing import Optional, List, Any, Dict


class QueryRequest(BaseModel):
    """API model for a query request."""
    query: str
    conversation_id: Optional[str] = None
    top_k: Optional[int] = 5


class ChoiceMessage(BaseModel):
    """API model for a message within a choice."""
    role: str
    content: str

class Choice(BaseModel):
    """API model for a single response choice."""
    index: int
    message: ChoiceMessage
    finish_reason: Optional[str] = None


class QueryResponse(BaseModel):
    """API model for a standard query response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[Dict[str, int]] = None
    conversation_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
