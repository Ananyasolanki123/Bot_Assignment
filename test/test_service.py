import pytest
from unittest.mock import patch, MagicMock
from src.Db.models import Message, MessageRole

from src.Services.llm_service import call_llm_api, manage_context_window, MAX_MODEL_TOKENS, CONTEXT_LIMIT 
from src.Services.llm_service import SYSTEM_PROMPT, count_tokens # Added for better context management assertions

def mock_groq_response():
    mock_choice = MagicMock()
    mock_choice.message.content = "This is a deterministic mock reply from the LLM."
    
    mock_usage = MagicMock()
    mock_usage.total_tokens = 50 
    
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage
    return mock_response

@pytest.mark.asyncio
@patch('src.Services.llm_service.client') 
async def test_llm_api_call_success(mock_client):
    
    mock_client.chat.completions.create.return_value = mock_groq_response()
    
    history = [
        Message(content="user query", role=MessageRole.USER, sequence_number=1)
    ]
    
    result = await call_llm_api(history)
    
    assert result['content'] == "This is a deterministic mock reply from the LLM."
    assert result['model'] == "llama3-8b-8192" 
    assert result['token_usage'] == 50
    mock_client.chat.completions.create.assert_called_once()
    

def test_context_window_management_full():
    MOCK_CONTENT = "A" * (100 * 4) 
    history = [
        Message(content=MOCK_CONTENT, role=MessageRole.USER, sequence_number=i)
        for i in range(1, 6)
    ]
    
    trimmed_history = manage_context_window(history, rag_context=None)
    
    assert len(trimmed_history) == 5
    assert trimmed_history[0].sequence_number == 1
    assert trimmed_history[-1].sequence_number == 5


def test_context_window_management_truncation():
    
    BASE_TOKENS = count_tokens(SYSTEM_PROMPT) 

    MOCK_MESSAGE_TOKENS = 1000
    MOCK_CONTENT = "A" * (MOCK_MESSAGE_TOKENS * 4) 
    
    history = [
        Message(content=MOCK_CONTENT, role=MessageRole.USER, sequence_number=i)
        for i in range(1, 11)
    ]
    
    with patch('src.Services.llm_service.MAX_MODEL_TOKENS', 3850):

        new_context_limit = int(3850 * 0.8)
        with patch('src.Services.llm_service.CONTEXT_LIMIT', new_context_limit):
            
            trimmed_history = manage_context_window(history, rag_context=None)
            
            assert len(trimmed_history) == 3 
            assert trimmed_history[0].sequence_number == 8
            assert trimmed_history[1].sequence_number == 9

            assert trimmed_history[2].sequence_number == 10
