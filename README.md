# BOT GPT Backend

A powerful enterprise conversational assistant built with FastAPI, capable of engaging in open chat and Retrieval-Augmented Generation (RAG) using uploaded PDF documents.

## Features

-   **Conversational AI**: Powered by Groq (Llama 3 models) for fast and accurate responses.
-   **RAG Capabilities**: Upload PDF documents and chat with them. The system extracts, chunks, and embeds content for context-aware answers.
-   **Context Management**: Intelligent sliding window mechanism to manage LLM token limits while retaining conversation history.
-   **Conversation Management**: Create, list, retrieve, and delete conversations.
-   **Document Management**: Upload, process, and link documents to specific conversations.

## Tech Stack

-   **Framework**: FastAPI
-   **Database**: PostgreSQL
-   **ORM**: SQLAlchemy
-   **LLM**: Groq API (Llama 3)
-   **Embeddings**: Sentence Transformers (`sentence-transformers/all-MiniLM-L6-v2`)
-   **Containerization**: Docker

## Prerequisites

-   Python 3.12+
-   PostgreSQL Database
-   Groq API Key

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Ananyasolanki123/Bot_Assignment.git
cd Bot_Assignment
```

### 2. Environment Configuration

Create a `.env` file in the root directory with the following variables:

```env
GROQ_API_KEY=your_groq_api_key
DB_NAME=your_db_name
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_HOST=localhost
DB_PORT=5432
```

### 3. Run Locally

Install dependencies:

```bash
pip install -r requirement.txt
```

Run the application:

```bash
uvicorn src.main:app --reload
```

The API will be available at `http://localhost:8000`.

### 4. Run with Docker

Build the Docker image:

```bash
docker build -t bot-gpt .
```

Run the container:

```bash
docker run -d -p 8001:8000 --env-file .env bot-gpt
```

## API Documentation

Once the server is running, you can access the interactive API documentation (Swagger UI) at:

`http://localhost:8000/docs`

### Key Endpoints

#### Documents
-   `POST /api/v1/conversations/documents`: Upload and process a PDF document.
-   `POST /api/v1/conversations/documents/link`: Link processed documents to a conversation.

#### Conversations
-   `POST /api/v1/conversations/`: Start a new conversation (Open Chat or RAG).
-   `GET /api/v1/conversations/`: List all conversations for the user.
-   `GET /api/v1/conversations/{conversation_id}`: Get conversation history.
-   `POST /api/v1/conversations/{conversation_id}/messages`: Send a message to an existing conversation.
-   `DELETE /api/v1/conversations/{conversation_id}`: Delete a conversation and its linked data.

## Project Structure

```
.
├── src/
│   ├── api/            # API Routes and Controllers
│   ├── Db/             # Database Models and Connection
│   ├── Services/       # Business Logic (LLM, RAG, Processor)
│   └── main.py         # Application Entry Point
├── Dockerfile
├── requirement.txt
```
