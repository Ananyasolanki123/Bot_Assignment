#Db/models.py
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Enum, Text
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from .__init__ import Base # Import Base from the same directory's init file

# --- Enumerations ---

class ConversationMode(enum.Enum):
    """Defines the operational mode of the conversation."""
    OPEN_CHAT = "OPEN_CHAT"
    RAG_CHAT = "RAG_CHAT"

class MessageRole(enum.Enum):
    """Defines the sender of the message in the conversation history."""
    USER = "USER"
    ASSISTANT = "ASSISTANT"
    SYSTEM = "SYSTEM"

class ProcessingStatus(enum.Enum):
    """Status of a document during chunking/embedding for RAG."""
    PENDING = "PENDING"
    CHUNKING = "CHUNKING"
    READY = "READY"
    FAILED = "FAILED"

# --- Core Entities ---

class User(Base):
    """Represents a user of the BOT GPT platform."""
    __tablename__ = "users"
    
    # Primary Key
    user_id = Column(String, primary_key=True) # Using String for UUIDs
    
    # Metadata
    email = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    conversations = relationship("Conversation", back_populates="user")
    uploaded_documents = relationship("Document", back_populates="user")

class Conversation(Base):
    """Represents a single, ongoing chat session."""
    __tablename__ = "conversations"
    
    # Primary Key
    conversation_id = Column(String, primary_key=True)
    
    # Foreign Key
    user_id = Column(String, ForeignKey("users.user_id"), index=True, nullable=False)
    
    # Core Data
    title = Column(String, default="New Chat")
    mode = Column(Enum(ConversationMode), default=ConversationMode.OPEN_CHAT, nullable=False)
    
    # Metadata for filtering/ordering
    token_count = Column(Integer, default=0) # Bonus: Total tokens used in this chat
    last_updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="conversations")
    
    # Messages in order, crucial for context construction
    messages = relationship("Message", back_populates="conversation", order_by="Message.sequence_number")
    
    # Documents linked to this RAG conversation
    linked_documents = relationship(
        "Document", 
        secondary="conversation_document_link", 
        back_populates="conversations"
    )

class Message(Base):
    """Represents a single turn (user or assistant message) in a conversation."""
    __tablename__ = "messages"
    
    # Primary Key
    message_id = Column(String, primary_key=True)
    
    # Foreign Key
    conversation_id = Column(
        String, 
        ForeignKey("conversations.conversation_id", ondelete="CASCADE"), # <--- THIS IS THE FIX
        index=True, 
        nullable=False
    )    
    # Core Data
    sequence_number = Column(Integer, index=True, nullable=False) # CRITICAL for ordering history
    role = Column(Enum(MessageRole), nullable=False)
    content = Column(Text, nullable=False)
    
    # Metadata
    llm_model = Column(String) # Bonus: Which model generated this reply
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    conversation = relationship("Conversation", back_populates="messages")

# --- RAG Entities ---

class Document(Base):
    __tablename__ = "documents"

    document_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"), index=True, nullable=False)
    filename = Column(String, nullable=False)
    storage_path = Column(String, nullable=False)
    processing_status = Column(Enum(ProcessingStatus), default=ProcessingStatus.PENDING, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="uploaded_documents")
    conversations = relationship(
        "Conversation",
        secondary="conversation_document_link",
        back_populates="linked_documents"
    )

    # ADD THIS RELATIONSHIP 👇
    chunks = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan"
    )

class ConvDocumentLink(Base):
    __tablename__ = "conversation_document_link"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String, ForeignKey("conversations.conversation_id"), index=True, nullable=False)
    document_id = Column(String, ForeignKey("documents.document_id"), index=True, nullable=False)


# ----- DOCUMENT CHUNK -----

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    chunk_id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey("documents.document_id"), index=True, nullable=False)

    content = Column(Text, nullable=False)
    embedding = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)

    document = relationship("Document", back_populates="chunks")