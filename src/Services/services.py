from sqlalchemy.orm import Session
from sqlalchemy import select
import uuid
from typing import List, Optional, Tuple, Dict, Any 
import logging
from fastapi import HTTPException # Needed for error propagation

# --- CORRECTED IMPORTS ---
# Assuming Db/models is the correct path for your ORM classes
from src.Db.models import User, Conversation, Message, MessageRole, ConversationMode 
from src.Services.rag_service import create_document_and_link
from src.Db.models import ConvDocumentLink, Document, DocumentChunk

from src.Services.llm_service import call_llm_api
from src.Services.rag_service import get_documents_for_conversation
from src.Services.rag_service import link_documents_to_conversation, retrieve_context_for_query
logger = logging.getLogger(__name__)

### --- Helper Function for Message Creation (Retained) --- ###

def _get_next_sequence_number(db: Session, conversation_id: str) -> int:
    """Finds the next sequence number for a new message in a conversation."""
    max_seq = db.query(Message.sequence_number).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.sequence_number.desc()).first()
    
    # If max_seq exists, get the integer value from the tuple, otherwise 0.
    return (max_seq[0] + 1) if max_seq else 1

### --- CRUD Functions --- ###

def create_initial_conversation(
    db: Session, 
    user_id: str, 
    first_message_content: str, 
    mode: str,
    document_ids: List[str] = None # CORRECTED: Added document_ids parameter
) -> Tuple[Optional[Conversation], Optional[Message]]:
    """
    Creates a new conversation and registers the user's first message, 
    including document linkage if in RAG mode.
    """
    # 1. Ensure User Exists (Simplified)
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        user = User(user_id=user_id, email=f"{user_id}@botgpt.com")
        db.add(user)
        db.flush()
        
    # 2. Create Conversation
    new_conversation = Conversation(
        conversation_id=str(uuid.uuid4()),
        user_id=user_id,
        mode=mode
    )
    db.add(new_conversation)
    db.flush()
    
    # 3. Link Documents if RAG Mode is requested (Correctly uses the new parameter)
    if mode == ConversationMode.RAG_CHAT.value and document_ids:
        # Call the RAG service function to create link entries
        link_documents_to_conversation(
            db=db, 
            conversation_id=new_conversation.conversation_id, 
            document_ids=document_ids
        )
        # Note: commit() happens in the linkage function for atomicity, but 
        # we still commit the conversation and message below.

    # 4. Create User's First Message (Sequence 1)
    user_message = Message(
        message_id=str(uuid.uuid4()),
        sequence_number=1,
        role=MessageRole.USER,
        content=first_message_content
    )
    new_conversation.messages.append(user_message)
    
    db.commit()
    db.refresh(new_conversation)
    db.refresh(user_message) 

    return new_conversation,user_message

def get_conversations_list(db: Session, user_id: str) -> List[Conversation]:
    """Retrieves a list of all conversations for a specific user."""
    conversations = db.query(Conversation).filter(
        Conversation.user_id == user_id
    ).order_by(Conversation.last_updated_at.desc()).all()
    
    return conversations

def get_conversation_detail(db: Session, conversation_id: str) -> Optional[Conversation]:
    """Retrieves a full conversation object, including all related messages."""
    # Use relationship loading for efficient retrieval of messages
    conversation = db.query(Conversation).filter(
        Conversation.conversation_id == conversation_id
    ).first()
    return conversation

def delete_conversation(db: Session, conversation_id: str):
    """Deletes a conversation and all its associated messages."""
    result = db.query(Conversation).filter(
        Conversation.conversation_id == conversation_id
    ).delete(synchronize_session=False) 
    
    db.commit()
    return result > 0

def add_user_message(db: Session, conversation_id: str, content: str) -> Optional[Message]:
    """Adds a new user message to an existing conversation."""
    
    next_seq = _get_next_sequence_number(db, conversation_id)
    
    user_message = Message(
        message_id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        sequence_number=next_seq,
        role=MessageRole.USER,
        content=content
    )
    db.add(user_message)
    db.commit()
    db.refresh(user_message)
    
    return user_message

# --- FINAL LLM ORCHESTRATION FUNCTION (Replaces the intermediate version) ---

async def process_user_message_and_get_reply(
    db: Session, 
    conversation_id: str,
    user_message_content: str,
    rag_context: Optional[str] = None

) -> Optional[Message]:
    """
    Handles a full conversation turn, including RAG retrieval if needed.
    """

    # 1. Fetch Conversation
    conversation = get_conversation_detail(db, conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    # 2. Save User Message
    user_message = add_user_message(db, conversation_id, user_message_content)
    db.refresh(conversation)  # ensure new message is in conversation.messages

    # 3. RAG Retrieval: Fetch linked documents and context
    rag_context = None
    linked_docs = []
    if conversation.mode == ConversationMode.RAG_CHAT:
        linked_docs = get_documents_for_conversation(db, conversation.conversation_id)
        
        if not linked_docs:
            logger.warning(f"No documents linked to conversation {conversation.conversation_id}")
        
        rag_context = retrieve_context_for_query(
            db=db,
            conversation=conversation,
            user_query=user_message_content
        )

        if not rag_context:
            logger.warning(f"RAG context empty for conversation {conversation.conversation_id}")
        else:
            logger.info(f"RAG context retrieved ({len(linked_docs)} docs): {rag_context[:50]}...")

    # 4. Collect Conversation History
    history = conversation.messages

    # 5. Call LLM API (with RAG context if available)
    llm_result = await call_llm_api(history, rag_context)

    # 6. Create and Save Assistant's Reply
    next_seq = _get_next_sequence_number(db, conversation_id)
    assistant_message = Message(
        message_id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        sequence_number=next_seq,
        role=MessageRole.ASSISTANT,
        content=llm_result['content'],
        llm_model=llm_result['model']
    )
    db.add(assistant_message)

    # 7. Update Conversation Metadata
    conversation.token_count += llm_result.get('token_usage', 0)

    db.commit()
    db.refresh(assistant_message)

    return assistant_message





# MOCK Assistant function included for any endpoints that might still call it (like a separate /mock endpoint)
def add_assistant_message_mock(db: Session, conversation_id: str, content: str) -> Optional[Message]:
    """Mocks the assistant's reply for the CRUD phase."""
    # This logic should be phased out once process_user_message_and_get_reply is fully used.
    max_seq = db.query(Message.sequence_number).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.sequence_number.desc()).first()
    next_seq = (max_seq[0] + 1) if max_seq else 1
    
    assistant_message = Message(
        message_id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        sequence_number=next_seq,
        role=MessageRole.ASSISTANT,
        content=content,
        llm_model="MOCK-GPT"
    )
    db.add(assistant_message)
    db.commit()
    db.refresh(assistant_message)
    
    return assistant_message