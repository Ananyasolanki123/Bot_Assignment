from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime, timedelta
import logging
import asyncio


from src.Db.__init__ import get_db
from src.Db.models import ConversationMode,MessageRole,ProcessingStatus,User,Conversation,Message,Document,ConvDocumentLink,DocumentChunk



import uuid
from src.Services.rag_service import delete_documents_for_conversation
from src.Services import rag_service
from src.Services import services as conversation_service
from src.Services import document_processor # <--- NEW IMPORT
from src.Db.Schema import (
    ConversationCreate,
    ConversationListResponse,
    ConversationResponse,
    ConversationContinue,
    MessageResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/conversations",
    tags=["Conversations"]
)

# Mock Auth
def get_current_user_id():
    return "user_mock_id_123"

# ----------------------------------------------------
# 1A. FILE UPLOAD ENDPOINT 
# ----------------------------------------------------
@router.post("/documents", status_code=status.HTTP_201_CREATED, tags=["Documents"])
async def upload_document(
    file: UploadFile = File(..., description="The PDF document to upload and process for RAG."),
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):

    if file.filename.split('.')[-1].lower() not in ["pdf"]:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Only PDF files are currently supported.")
    
    try:
        document_id = await document_processor.process_document_upload(db, user_id, None, file)

        
        if user_id not in pending_documents_store:
            pending_documents_store[user_id] = []
            
        timestamp = datetime.utcnow()
        pending_documents_store[user_id].append((document_id, timestamp))
        
        return {
            "message": "Document uploaded and processing complete. ID saved for linking.",
            "document_id": document_id,
            "filename": file.filename
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Uncaught exception during file upload: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal server error occurred during processing: {str(e)}"
        )

# ----------------------------------------------------
# 1B. Link Documents by ID 
# ----------------------------------------------------

@router.post("/documents/link", status_code=status.HTTP_201_CREATED, tags=["Documents"])
def link_docs_to_conversation(
    document_ids: List[str],
    conversation_id: str = None,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    if conversation_id:
        conversation = conversation_service.get_conversation_detail(db, conversation_id)
        if not conversation or conversation.user_id != user_id:
            raise HTTPException(404, "Conversation not found or unauthorized access.")

        rag_service.link_documents_to_conversation(
            db=db,
            conversation_id=conversation_id,
            document_ids=document_ids
        )

        return {"message": "Documents linked successfully", "document_ids": document_ids}

    else:
       
        if user_id not in pending_documents_store:
            pending_documents_store[user_id] = []

        timestamp = datetime.utcnow()
        for doc_id in document_ids:
            pending_documents_store[user_id].append((doc_id, timestamp))

        return {
            "message": "Documents stored temporarily. They will be linked when a conversation starts.",
            "document_ids": document_ids
        }

# ----------------------------------------------------
# 2. Start New Conversation 
# ----------------------------------------------------
@router.post(
    "/",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED
)
async def start_new_conversation(
    payload: ConversationCreate,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
   
    if not payload.first_message or payload.first_message.strip() == "":
        raise HTTPException(status_code=400, detail="First message cannot be empty.")

    # 1️ Create the conversation
    conversation, _ = conversation_service.create_initial_conversation(
        db=db,
        user_id=user_id,
        first_message_content=payload.first_message,
        mode=payload.mode.value,
        document_ids=None  # link manually below
    )

    # 2️ Collect all documents to link
    all_doc_ids = []

    # 2a. Documents provided in payload
    if payload.document_ids:
        for doc_id in payload.document_ids:
            doc = db.query(Document).filter(Document.document_id == doc_id).first()
            if doc:
                # Wait briefly if processing is not yet ready
                for _ in range(10):  # retry 10 times
                    if doc.processing_status == ProcessingStatus.READY:
                        all_doc_ids.append(doc_id)
                        break
                    await asyncio.sleep(0.5)  # 0.5s wait
                    db.refresh(doc)

    # 2b. Pending documents uploaded before conversation start
    pending_docs = pending_documents_store.get(user_id, [])
    if pending_docs:
        cutoff = datetime.utcnow() - timedelta(days=7)
        recent_docs = [doc_id for doc_id, ts in pending_docs if ts >= cutoff]

        for doc_id in recent_docs:
            if doc_id in all_doc_ids:
                continue  # already included
            doc = db.query(Document).filter(Document.document_id == doc_id).first()
            if doc:
                for _ in range(10):
                    if doc.processing_status == ProcessingStatus.READY:
                        all_doc_ids.append(doc_id)
                        break
                    await asyncio.sleep(0.5)
                    db.refresh(doc)

        # Remove old or expired docs from pending store
        pending_documents_store[user_id] = [
            (doc_id, ts) for doc_id, ts in pending_docs if ts < cutoff
        ]

    # 3️ Link all READY documents to conversation
    if all_doc_ids:
        rag_service.link_documents_to_conversation(
            db=db,
            conversation_id=conversation.conversation_id,
            document_ids=all_doc_ids
        )

    db.commit()  # save all document links

    # 4️ Fetch linked documents for RAG
    linked_docs = rag_service.get_documents_for_conversation(db, conversation.conversation_id)

    # 5️ Generate assistant reply (pass documents if RAG_CHAT)
    # Note: Service handles RAG context retrieval internally based on conversation mode
    assistant_message = await conversation_service.process_user_message_and_get_reply(
        db=db,
        conversation_id=conversation.conversation_id,
        user_message_content=payload.first_message
    )

    # 6️ Fetch final conversation
    final_conversation = conversation_service.get_conversation_detail(
        db, conversation.conversation_id
    )

    if not final_conversation:
        raise HTTPException(500, "Failed to retrieve conversation after processing.")

    return final_conversation
# ----------------------------------------------------
# 3. List Conversations
# ----------------------------------------------------
@router.get("/", response_model=List[ConversationListResponse])
def list_conversations(
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    conversations = conversation_service.get_conversations_list(db=db, user_id=user_id)
    return conversations

# ----------------------------------------------------
# 4. Get Conversation History
# ----------------------------------------------------
@router.get("/{conversation_id}", response_model=ConversationResponse)
def get_conversation_history(
    conversation_id: str,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    conversation = conversation_service.get_conversation_detail(
        db=db, conversation_id=conversation_id
    )

    if not conversation or conversation.user_id != user_id:
        raise HTTPException(404, "Conversation not found or unauthorized access.")

    return conversation

# ----------------------------------------------------
# 5. Delete Conversation (with all related documents)
# ----------------------------------------------------
@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: str,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    conversation = conversation_service.get_conversation_detail(
        db=db, conversation_id=conversation_id
    )

    if not conversation or conversation.user_id != user_id:
        raise HTTPException(404, "Conversation not found or unauthorized access.")

    # Delete all documents linked to this conversation
    rag_service.delete_documents_for_conversation(db=db, conversation_id=conversation_id)

    # Delete conversation and messages
    conversation_service.delete_conversation(db=db, conversation_id=conversation_id)

    return None

# ----------------------------------------------------
# 6. Continue Conversation (auto-considers newly linked documents)
# ----------------------------------------------------
@router.post("/{conversation_id}/messages", response_model=MessageResponse)
async def continue_conversation(
    conversation_id: str,
    payload: ConversationContinue,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    conversation = conversation_service.get_conversation_detail(
        db=db, conversation_id=conversation_id
    )

    if not conversation or conversation.user_id != user_id:
        raise HTTPException(404, "Conversation not found or unauthorized access.")

    linked_docs = rag_service.get_documents_for_conversation(db, conversation_id)

    # Unified processor handles both RAG and normal chat
    assistant_reply = await conversation_service.process_user_message_and_get_reply(
        db=db,
        conversation_id=conversation_id,
        user_message_content=payload.user_message
    )


    return assistant_reply  # <-- ORM Message mapped via Pydantic model
