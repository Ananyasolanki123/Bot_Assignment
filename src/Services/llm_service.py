import os
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from groq import Groq, APIError
from fastapi import HTTPException, status
from dotenv import load_dotenv

from src.Db.models import Message, MessageRole 

load_dotenv()

logger = logging.getLogger(__name__)

LLM_MODEL = "llama-3.1-8b-instant" 
MAX_MODEL_TOKENS = 32768 # Updated context window size for Mixtral
SAFETY_THRESHOLD = 0.8  # Use 80% of total tokens for safety margin
CONTEXT_LIMIT = int(MAX_MODEL_TOKENS * SAFETY_THRESHOLD) # Calculation updated: 26214 tokens
SYSTEM_PROMPT = (
    "You are BOT GPT, a helpful and concise enterprise conversational assistant. "
    "Your goal is to answer user queries based on conversation history and provided documents. "
    "Be professional and brief."
)
# ---------------------

# Initialize the Groq client
try:
    client = Groq() 
except Exception as e:
    client = None
    logger.error(f"Failed to initialize Groq Client: {e}. Check your GROQ_API_KEY.")

# --- UTILITY FUNCTIONS ---

def count_tokens(text: str) -> int:

    if not text:
        return 0
    # Common approximation: 4 characters per token
    return len(text) // 4

def format_messages_for_llm(
    conversation_history: List[Message], 
    rag_context: Optional[str] = None
) -> List[Dict[str, str]]:
  
    formatted_messages = []
    
    full_system_prompt = SYSTEM_PROMPT
    if rag_context:
        full_system_prompt = (
            f"RAG CONTEXT:\n---\n{rag_context}\n---\n\n"
            f"{SYSTEM_PROMPT}"
        )
        
    formatted_messages.append({"role": "system", "content": full_system_prompt})

    for message in conversation_history:
        role_str = message.role.value.lower() 
        formatted_messages.append({
            "role": role_str, 
            "content": message.content
        })

    return formatted_messages


def manage_context_window(
    conversation_history: List[Message], 
    rag_context: Optional[str] = None
) -> List[Message]:

    
    base_prompt = SYSTEM_PROMPT
    base_tokens = count_tokens(base_prompt)
    if rag_context:
        base_tokens += count_tokens(rag_context)

    
    current_tokens = base_tokens
    processed_history: List[Message] = []
    
    # Iterate backwards through the history (newest message first)
    for message in reversed(conversation_history):
        message_tokens = count_tokens(message.content)
        
        # Check if adding this message exceeds the limit
        if current_tokens + message_tokens <= CONTEXT_LIMIT:
            current_tokens += message_tokens
            
            processed_history.insert(0, message)
        else:
            logger.warning(
                f"Context window hit limit ({CONTEXT_LIMIT} tokens). "
                f"Discarding message sequence #{message.sequence_number}."
            )
            break
            

    return processed_history

# ---------------------

async def call_llm_api(conversation_history: List[Message], rag_context: Optional[str] = None) -> Dict[str, Any]:

    if client is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Groq Client is not initialized. Check server logs for API key errors."
        )

    # 1. Manage Context Window (Sliding Window applied)
    processed_history = manage_context_window(conversation_history, rag_context)
    
    # 2. Format Payload for the Groq API
    messages_payload = format_messages_for_llm(processed_history, rag_context)

    # --- LLM API CALL AND ERROR HANDLING ---
    
    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            # Groq API Call
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages_payload,
                # Setting temperature to 0 for consistent, deterministic enterprise answers
                temperature=0 
            )

            # 3. Parse and Store Response
            if response.choices:
                # Get the text content
                content = response.choices[0].message.content
                
                # Retrieve token usage (Cost tracking)
                total_tokens = response.usage.total_tokens if response.usage else 0
                
                return {
                    'content': content,
                    'model': LLM_MODEL,
                    'token_usage': total_tokens
                }
            else:
                raise Exception("Groq returned an empty response candidate list.")
        
        except APIError as e:
            # Handle Groq-specific API errors (e.g., Rate Limits, Invalid Key)
            logger.error(f"Groq API Error (Status {e.status_code}) on attempt {attempt + 1}: {e}")
            
            # Use exponential backoff for retries
            if attempt < MAX_RETRIES - 1:
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
            else:
                # 503 Fail-safe Response 
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="External Groq service is unavailable after multiple retries."
                )
        
        except Exception as e:
            # Catch all other exceptions (e.g., network, client issues)
            logger.error(f"General LLM API Call failed on attempt {attempt + 1}: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="External Groq service failed due to an unknown error."
            )

    # Fallback exception if loop finishes without success
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="LLM service failed after all retries."

    )
