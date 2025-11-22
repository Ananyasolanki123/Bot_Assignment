from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

from src.Db.models import MessageRole, ConversationMode 

# ---------------------------------------------
# 1. Message Schemas (Base, Input, and Output)
# ---------------------------------------------

class MessageBase(BaseModel):
    """Base structure for data contained in a message object."""
    content: str
    role: MessageRole # Uses the Enum from models.py

class MessageResponse(MessageBase):
    """Schema for returning a message in API responses (includes DB metadata)."""
    message_id: str
    sequence_number: int
    created_at: datetime
    llm_model: Optional[str] = None # Bonus field

    class Config:
        # Crucial setting: Allows Pydantic to read data from the ORM objects
        # using attribute access (e.g., conversation.messages) instead of dict keys.
        from_attributes = True 

# ---------------------------------------------
# 2. Conversation Schemas (Input)
# ---------------------------------------------

class ConversationCreate(BaseModel):
    """Input payload for POST /conversations (Starting a new chat)."""
    first_message: str = Field(..., min_length=1, description="The user's initial message.")
    
    # Defaults to OPEN_CHAT, but allows the user to specify RAG_CHAT mode
    mode: ConversationMode = ConversationMode.OPEN_CHAT
    
    # Optional list of Document IDs if starting a Grounded/RAG chat
    document_ids: List[str] = Field(default_factory=list, description="List of Document IDs to ground the conversation.")

class ConversationContinue(BaseModel):
    """Input payload for POST /conversations/{id}/messages (Continuing a chat)."""
    user_message: str = Field(..., min_length=1, description="The user's subsequent message.")

# ---------------------------------------------
# 3. Conversation Schemas (Output)
# ---------------------------------------------

class ConversationListResponse(BaseModel):
    """Simplified schema for GET /conversations (Listing past chats)."""
    conversation_id: str
    title: str
    last_updated_at: datetime
    
    class Config:
        from_attributes = True

class ConversationResponse(BaseModel):
    """Detailed response schema for GET /conversations/{id} (Full history)."""
    conversation_id: str
    user_id: str
    title: str
    mode: ConversationMode
    last_updated_at: datetime
    
    # Bonus field for cost tracking
    token_count: int
    
    # The list of messages, serialized using the MessageResponse schema
    messages: List[MessageResponse] = [] 
    
    class Config:

        from_attributes = True
