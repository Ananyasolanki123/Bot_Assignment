import requests
import os

# Assuming backend runs on 8000
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1/conversations")

def upload_document(file_obj):
    """Uploads a single PDF document to the backend."""
    url = f"{API_BASE_URL}/documents"
    files = {'file': (file_obj.name, file_obj, 'application/pdf')}
    try:
        response = requests.post(url, files=files)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error uploading document: {e}")
        try:
            print(f"Response status: {response.status_code}, body: {response.text}")
        except:
            pass
        return None

def link_docs_to_conversation(document_ids, conversation_id=None):
    """Links uploaded documents to a conversation (or stores them as pending)."""
    url = f"{API_BASE_URL}/documents/link"
    data = {"document_ids": document_ids}
    if conversation_id:
        url += f"?conversation_id={conversation_id}"
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error linking documents: {e}")
        return None

def start_new_conversation(first_message, mode="OPEN_CHAT", document_ids=None):
    """Starts a new conversation."""
    url = f"{API_BASE_URL}/"
    payload = {
        "first_message": first_message,
        "mode": mode,
        "document_ids": document_ids or []
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error starting conversation: {e}")
        try:
            print(f"Response status: {response.status_code}, body: {response.text}")
        except:
            pass
        return None

def list_conversations():
    """Lists all conversations for the user."""
    url = f"{API_BASE_URL}/"
    try:
        response = requests.get(url)
        response.raise_for_status()
        if not response.text:
            return []
        return response.json()
    except Exception as e:
        print(f"Error listing conversations: {e}")
        try:
            print(f"Response status: {response.status_code}, body: {response.text}")
        except:
            pass
        return []

def get_conversation_history(conversation_id):
    """Gets detailed history for a specific conversation."""
    url = f"{API_BASE_URL}/{conversation_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error getting conversation history: {e}")
        return None

def continue_conversation(conversation_id, message):
    """Sends a new message to an existing conversation."""
    url = f"{API_BASE_URL}/{conversation_id}/messages"
    payload = {"user_message": message}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error continuing conversation: {e}")
        try:
            print(f"Response status: {response.status_code}, body: {response.text}")
        except:
            pass
        return None

def delete_conversation(conversation_id):
    """Deletes a conversation."""
    url = f"{API_BASE_URL}/{conversation_id}"
    try:
        response = requests.delete(url)
        if response.status_code == 404:
            print(f"Conversation {conversation_id} already deleted or not found.")
            return True
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Error deleting conversation: {e}")
        return False
