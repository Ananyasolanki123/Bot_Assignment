#Db/__init__.py

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
from urllib.parse import quote_plus # <--- NEW IMPORT

# 1. Environment Setup and URL Construction
load_dotenv()

# Extract individual components from the .env file
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_PASSWORD_ENCODED = quote_plus(DB_PASSWORD) # <--- CRUCIAL STEP
# Construct the required SQLAlchemy database URL (using psycopg2 driver)
SQLALCHEMY_DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD_ENCODED}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# 2. Engine Creation 
# Note: For production use, you should handle the case where any ENV var is missing.
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    pool_recycle=3600
)

# 3. Session Setup 
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 4. Base Class 
Base = declarative_base()
from . import models

# 5. Dependency 
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Helper function to create all tables (used for initial setup/migrations)
def create_db_tables():
    # Base.metadata.create_all(bind=engine) will create all tables 
    # defined in models.py that inherit from Base.
    Base.metadata.create_all(bind=engine)