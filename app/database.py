from pymongo import MongoClient
from app.config import MONGODB_URL, DATABASE_NAME

client = MongoClient(MONGODB_URL)
db = client[DATABASE_NAME]

# Collections
users_collection = db["users"]
chatbots_collection = db["chatbots"]
conversations_collection = db["conversations"]
documents_collection = db["documents"]

# Create indexes
users_collection.create_index("username", unique=True)
users_collection.create_index("email", unique=True)
chatbots_collection.create_index("user_id")
documents_collection.create_index("chatbot_id") 