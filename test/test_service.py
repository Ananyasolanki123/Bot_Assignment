import pytest
from unittest.mock import patch, MagicMock
from src.Db.models import Message, MessageRole

# CORRECTED IMPORT PATH: Use the absolute path starting from the top-level package 'src'
from src.Services.llm_service import call_llm_api, manage_context_window, MAX_MODEL_TOKENS, CONTEXT_LIMIT 
from src.Services.llm_service import SYSTEM_PROMPT, count_tokens # Added for better context management assertions

# --- Mock Response Object ---
def mock_groq_response():
    """Creates a mock response object matching the Groq SDK's structure."""
    mock_choice = MagicMock()
    mock_choice.message.content = "This is a deterministic mock reply from the LLM."
    
    mock_usage = MagicMock()
    mock_usage.total_tokens = 50 
    
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage
    return mock_response

@pytest.mark.asyncio
# CORRECTED PATCH PATH: Use the absolute path
@patch('src.Services.llm_service.client') 
async def test_llm_api_call_success(mock_client):
    """Tests if the LLM call succeeds and parses the response correctly."""
    
    # Configure the mock client's create method to return our predefined response
    mock_client.chat.completions.create.return_value = mock_groq_response()
    
    # Create mock history for the test
    history = [
        Message(content="user query", role=MessageRole.USER, sequence_number=1)
    ]
    
    # Act: Call the service function
    result = await call_llm_api(history)
    
    # Assert: Verify the output structure and mocked data
    assert result['content'] == "This is a deterministic mock reply from the LLM."
    assert result['model'] == "llama3-8b-8192" 
    assert result['token_usage'] == 50
    mock_client.chat.completions.create.assert_called_once()
    
# --- Context Window Tests ---

def test_context_window_management_full():
    """Tests if all messages are included when context limit is not exceeded."""
    # Create 5 mock messages, each simulated to be 100 tokens (500 total)
    # The default CONTEXT_LIMIT (6553 tokens) will easily accommodate this.
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
    """Tests the Sliding Window logic when the limit is hit."""
    
    # Base prompt tokens (SYSTEM_PROMPT is about 40 tokens)
    BASE_TOKENS = count_tokens(SYSTEM_PROMPT) 

    # Mock the content size to exactly 1000 tokens per message
    MOCK_MESSAGE_TOKENS = 1000
    MOCK_CONTENT = "A" * (MOCK_MESSAGE_TOKENS * 4) 
    
    # Create 10 mock messages
    history = [
        Message(content=MOCK_CONTENT, role=MessageRole.USER, sequence_number=i)
        for i in range(1, 11)
    ]
    
    # Patch the max tokens to allow only 3 full messages (3000 tokens) + Base
    # The context limit is 0.8 * MAX_MODEL_TOKENS. 
    # To fit exactly 3 messages, we need: (3 * 1000) + BASE_TOKENS. Let's aim for 4000.
    
    # New Max Model Tokens = (3 * 1000) / 0.8 + 100 (approx) = 3850
    # CONTEXT_LIMIT = 3850 * 0.8 = 3080. This safely fits 3 messages (3000 tokens)
    with patch('src.Services.llm_service.MAX_MODEL_TOKENS', 3850):
        # Must re-calculate CONTEXT_LIMIT for the patched token count
        # and mock the module-level variable
        new_context_limit = int(3850 * 0.8)
        with patch('src.Services.llm_service.CONTEXT_LIMIT', new_context_limit):
            
            # Act: Run the context manager
            trimmed_history = manage_context_window(history, rag_context=None)
            
            # Assert: Expect only the 3 most recent messages (Sequences 8, 9, 10)
            assert len(trimmed_history) == 3 
            assert trimmed_history[0].sequence_number == 8
            assert trimmed_history[1].sequence_number == 9
            assert trimmed_history[2].sequence_number == 10