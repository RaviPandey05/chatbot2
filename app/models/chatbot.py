from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class ChatbotBase(BaseModel):
    name: str = "New Chatbot"
    description: str = "Default chatbot description"
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    temperature: float = 0.7
    system_prompt: str = """You are a helpful AI assistant. Please format your responses as follows:

📌 **Answer:**
[Provide a clear, concise answer]

📌 **Explanation:**
- [Key point 1]
- [Key point 2]
- [Key point 3]

If relevant context is available:
📌 **From the Context:**
[Include relevant information from the provided context]

📌 **Follow-up Suggestion:**
[Suggest a related topic or question]"""

    prompt_template: str = """System: {system_prompt}

Context: {context}

Question: {query}

Remember to follow the response format specified in the system prompt."""
    pinecone_namespace: str = ""

class ChatbotCreate(BaseModel):
    name: str
    description: str = "Default description"
    user_id: Optional[str] = None
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    temperature: float = 0.7
    pinecone_namespace: str = ""

class ChatbotUpdate(BaseModel):
    name: str
    description: str
    temperature: float

class ChatbotInDB(ChatbotBase):
    id: str
    user_id: str
    created_at: datetime
    documents: List[str] = []
    urls: List[str] = []
    pinecone_namespace: str

    class Config:
        from_attributes = True

class DocumentMetadata(BaseModel):
    filename: str
    doc_type: str
    upload_date: datetime
    processed: bool = False
    embedding_count: int = 0 