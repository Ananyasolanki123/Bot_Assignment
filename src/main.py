
from fastapi import FastAPI
# Import the router you just finalized
from src.api.router import router as conversation_router
# Import necessary dependencies (for setup)
from src.Db.__init__ import create_db_tables, get_db
from src.Db import models # Ensures ORM models are registered

# Initialize FastAPI app
app = FastAPI(
    title="BOT GPT Backend", 
    version="v1.0.0"
)

# Include the router under the API version prefix
app.include_router(conversation_router, prefix="/api/v1")

