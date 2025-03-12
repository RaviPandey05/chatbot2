import os
from dotenv import load_dotenv
from pinecone import Pinecone
from datetime import datetime

load_dotenv()

# API Keys and External Services
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# MongoDB Configuration
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "chatbot_platform")

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Pinecone Configuration
PINECONE_ENVIRONMENT = "us-east-1-aws"
INDEX_NAME = "chatbot-index-1"
EMBEDDING_DIM = 384

pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialize Pinecone index if it doesn't exist
existing_indexes = pc.list_indexes()
if INDEX_NAME not in existing_indexes:
    try:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec={
                "serverless": {
                    "cloud": "aws",
                    "region": "us-east-1"
                }
            }
        )
        print(f"Created new Pinecone index: {INDEX_NAME}")
    except Exception as e:
        print(f"Using existing index: {INDEX_NAME}")
        pass

index = pc.Index(INDEX_NAME)

# Model Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2" 