import uvicorn
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.security import OAuth2PasswordRequestForm
from bson import ObjectId
from app.services.auth import (
    create_user,
    authenticate_user,
    create_access_token,
    get_current_user
)
from app.models.user import User, UserCreate
from app.services.chatbot import ChatbotService
from app.models.chatbot import ChatbotCreate, ChatbotUpdate, ChatbotInDB
from app.services.document_processor import DocumentProcessor
from fastapi.middleware.cors import CORSMiddleware
from app.database import chatbots_collection
import magic

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication routes
@app.post("/api/auth/register")
async def register(user: UserCreate):
    return await create_user(user)

@app.post("/api/auth/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

# Initialize services
chatbot_service = ChatbotService()
document_processor = DocumentProcessor()

# Chatbot routes
@app.get("/api/chatbots")
async def get_chatbots(current_user: User = Depends(get_current_user)):
    chatbots = list(chatbots_collection.find({"user_id": current_user.username}))
    # Convert ObjectId to string for JSON serialization
    for chatbot in chatbots:
        chatbot["id"] = str(chatbot["_id"])
        del chatbot["_id"]
    return chatbots

@app.post("/api/chatbots", status_code=201)
async def create_chatbot(
    chatbot: ChatbotCreate,
    current_user: User = Depends(get_current_user)
):
    # Set the user_id from the authenticated user
    chatbot.user_id = current_user.username
    return await chatbot_service.create_chatbot(chatbot)

@app.post("/api/chatbots/{chatbot_id}/chat")
async def chat(
    chatbot_id: str, 
    message: dict, 
    current_user: User = Depends(get_current_user)
):
    # Verify chatbot ownership
    chatbot = chatbots_collection.find_one({"_id": ObjectId(chatbot_id)})
    if not chatbot or chatbot["user_id"] != current_user.username:
        raise HTTPException(status_code=404, detail="Chatbot not found")
    
    response = await chatbot_service.get_response(chatbot_id, message["text"])
    return response

@app.post("/api/chatbots/{chatbot_id}/documents")
async def upload_document(
    chatbot_id: str, 
    file: UploadFile = File(...), 
    current_user: User = Depends(get_current_user)
):
    # Verify chatbot ownership
    chatbot = chatbots_collection.find_one({"_id": ObjectId(chatbot_id)})
    if not chatbot or chatbot["user_id"] != current_user.username:
        raise HTTPException(status_code=404, detail="Chatbot not found")

    # Initialize document processor
    document_processor = DocumentProcessor()

    # Check if file type is supported
    if file.content_type not in document_processor.SUPPORTED_MIME_TYPES:
        try:
            # Read file content for type detection
            content = await file.read()
            # Use python-magic to detect actual file type
            detected_mime_type = magic.from_buffer(content, mime=True)
            
            if detected_mime_type not in document_processor.SUPPORTED_MIME_TYPES:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file type: {detected_mime_type}. Supported types are PDF, TXT, and DOCX."
                )
            
            # Process the file with detected mime type
            result = await document_processor.process_file(
                file_content=content,
                filename=file.filename,
                mime_type=detected_mime_type,
                chatbot_id=chatbot_id,
                namespace=chatbot["pinecone_namespace"]
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")
    else:
        # Process the file with original mime type
        content = await file.read()
        result = await document_processor.process_file(
            file_content=content,
            filename=file.filename,
            mime_type=file.content_type,
            chatbot_id=chatbot_id,
            namespace=chatbot["pinecone_namespace"]
        )

    return result

@app.post("/api/chatbots/{chatbot_id}/urls")
async def process_url(
    chatbot_id: str,
    url_data: dict,
    current_user: User = Depends(get_current_user)
):
    # Verify chatbot ownership
    chatbot = chatbots_collection.find_one({"_id": ObjectId(chatbot_id)})
    if not chatbot or chatbot["user_id"] != current_user.username:
        raise HTTPException(status_code=404, detail="Chatbot not found")

    if "url" not in url_data:
        raise HTTPException(400, "URL not provided")

    result = await document_processor.process_url(
        url=url_data["url"],
        chatbot_id=chatbot_id,
        namespace=chatbot["pinecone_namespace"]
    )
    return result

@app.put("/api/chatbots/{chatbot_id}")
async def update_chatbot(
    chatbot_id: str,
    update_data: ChatbotUpdate,
    current_user: User = Depends(get_current_user)
):
    # Verify chatbot ownership
    chatbot = chatbots_collection.find_one({"_id": ObjectId(chatbot_id)})
    if not chatbot or chatbot["user_id"] != current_user.username:
        raise HTTPException(status_code=404, detail="Chatbot not found")
    
    return await chatbot_service.update_chatbot(
        chatbot_id=chatbot_id,
        name=update_data.name,
        description=update_data.description,
        temperature=update_data.temperature,
        system_prompt=update_data.system_prompt,
        prompt_template=update_data.prompt_template
    )

@app.get("/api/chatbots/{chatbot_id}")
async def get_chatbot(
    chatbot_id: str,
    current_user: User = Depends(get_current_user)
):
    try:
        chatbot = chatbots_collection.find_one({"_id": ObjectId(chatbot_id)})
        if not chatbot or chatbot["user_id"] != current_user.username:
            raise HTTPException(status_code=404, detail="Chatbot not found")
        
        # Convert ObjectId to string for JSON serialization
        chatbot["id"] = str(chatbot["_id"])
        del chatbot["_id"]
        
        return chatbot
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chatbot: {str(e)}")

@app.get("/api/chatbots/{chatbot_id}/storage")
async def check_storage(
    chatbot_id: str,
    current_user: User = Depends(get_current_user)
):
    """Check document storage status for a chatbot."""
    chatbot = chatbots_collection.find_one({"_id": ObjectId(chatbot_id)})
    if not chatbot or chatbot["user_id"] != current_user.username:
        raise HTTPException(status_code=404, detail="Chatbot not found")
    
    return await chatbot_service.verify_document_storage(chatbot_id)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 