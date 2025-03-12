from typing import List, Optional
import httpx
from app.models.chatbot import ChatbotCreate, ChatbotInDB, ChatbotUpdate
from app.database import chatbots_collection
from app.config import (
    HUGGINGFACE_API_KEY, 
    EMBEDDING_MODEL, 
    PINECONE_API_KEY,
    INDEX_NAME,
    PINECONE_ENVIRONMENT
)
from datetime import datetime
import uuid
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from bson import ObjectId
from fastapi import HTTPException
import logging
import re

logger = logging.getLogger(__name__)

class ChatbotService:
    def __init__(self):
        self.headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        self.api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    async def create_chatbot(self, chatbot: ChatbotCreate) -> ChatbotInDB:
        # Generate a unique namespace for the chatbot
        namespace = f"chatbot_{str(uuid.uuid4())}"
        
        # Get user_id from the request or use default
        user_id = chatbot.user_id or "default_user"
        
        db_chatbot = {
            "name": chatbot.name,
            "description": chatbot.description,
            "user_id": user_id,
            "model_name": chatbot.model_name,
            "temperature": chatbot.temperature,
            "created_at": datetime.utcnow(),
            "documents": [],
            "urls": [],
            "pinecone_namespace": namespace
        }
        
        try:
            result = chatbots_collection.insert_one(db_chatbot)
            db_chatbot["id"] = str(result.inserted_id)
            return ChatbotInDB(**db_chatbot)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create chatbot: {str(e)}")

    async def update_chatbot(
        self,
        chatbot_id: str,
        name: str,
        description: str,
        temperature: float,
        system_prompt: str,
        prompt_template: str
    ) -> ChatbotInDB:
        try:
            result = chatbots_collection.find_one_and_update(
                {"_id": ObjectId(chatbot_id)},
                {"$set": {
                    "name": name,
                    "description": description,
                    "temperature": temperature,
                    "system_prompt": system_prompt,
                    "prompt_template": prompt_template,
                    "updated_at": datetime.utcnow()
                }},
                return_document=True
            )
            if result:
                result["id"] = str(result["_id"])
                del result["_id"]
                return ChatbotInDB(**result)
            raise HTTPException(status_code=404, detail="Chatbot not found")
        except Exception as e:
            logger.error(f"Error updating chatbot: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to update chatbot: {str(e)}")

    async def _get_relevant_context(self, query: str, namespace: str) -> str:
        try:
            logger.info(f"Getting context for query: {query} in namespace: {namespace}")
            
            # Create query embedding
            query_embedding = self.embedding_model.encode(query)
            
            # Query Pinecone
            index = self.pc.Index(INDEX_NAME)
            results = index.query(
                vector=query_embedding.tolist(),
                top_k=5,
                namespace=namespace,
                include_metadata=True,
                score_threshold=0.2  # Lower threshold for better recall
            )
            
            logger.info(f"Found {len(results.matches)} matches in Pinecone")
            
            if not results.matches:
                logger.warning(f"No matches found in namespace {namespace}")
                return "No relevant context found."

            # Filter and sort matches by score
            relevant_contexts = []
            for match in results.matches:
                if match.metadata and "text" in match.metadata:
                    # Clean the text
                    clean_text = re.sub(r'\s+', ' ', match.metadata["text"])
                    relevant_contexts.append({
                        'text': clean_text,
                        'score': match.score,
                        'source': match.metadata.get('source', 'unknown')
                    })
            
            if not relevant_contexts:
                return "No text content found in matches."

            # Sort by score and combine contexts
            relevant_contexts.sort(key=lambda x: x['score'], reverse=True)
            context_texts = []
            for ctx in relevant_contexts[:3]:
                source_text = f"\nSource: {ctx['source']}" if ctx['source'] != 'unknown' else ""
                context_texts.append(f"{ctx['text']}{source_text}")
            
            combined_context = "\n---\n".join(context_texts)
            logger.info(f"Retrieved context length: {len(combined_context)}")
            
            return combined_context

        except Exception as e:
            logger.error(f"Error getting context: {str(e)}")
            return f"Error retrieving context: {str(e)}"

    async def get_response(self, chatbot_id: str, query: str) -> dict:
        try:
            chatbot = chatbots_collection.find_one({"_id": ObjectId(chatbot_id)})
            if not chatbot:
                raise HTTPException(status_code=404, detail="Chatbot not found")

            context = await self._get_relevant_context(query, chatbot["pinecone_namespace"])
            
            if context in ["No relevant context found.", "No text content found in matches.", "Error retrieving context."]:
                return {
                    "response": "I don't have information about that in my knowledge base.",
                    "context_used": False,
                    "original_query": query
                }

            # Simple prompt that focuses on accurate information from context
            prompt = f"""Answer this question using ONLY the information provided in the context below. If the information isn't in the context, say "I don't have that information."

Context: {context}

Question: {query}

Answer:"""

            async with httpx.AsyncClient(timeout=45.0) as client:
                response = await client.post(
                    self.api_url,
                    headers=self.headers,
                    json={
                        "inputs": prompt,
                        "parameters": {
                            "temperature": 0.1,  # Lower temperature for more factual responses
                            "max_length": 512,
                            "top_p": 0.9
                        }
                    }
                )
                
                bot_response = response.json()[0]["generated_text"]
                
                # Clean up the response
                cleaned_response = self._clean_response(bot_response)
                
                # Verify the response uses context
                if not self._verify_response_uses_context(cleaned_response, context):
                    return {
                        "response": "I don't have specific information about that in my knowledge base.",
                        "context_used": False,
                        "original_query": query
                    }
                
                return {
                    "response": cleaned_response,
                    "context_used": True,
                    "original_query": query
                }

        except Exception as e:
            logger.error(f"Error in get_response: {str(e)}")
            raise HTTPException(status_code=500, detail="An error occurred while processing your request")

    def _verify_response_uses_context(self, response: str, context: str) -> bool:
        """Verify that the response is actually using information from the context."""
        response_text = response.lower()
        context_text = context.lower()
        
        # If the response indicates no information, check if that's accurate
        if any(phrase in response_text for phrase in ["i don't have", "no information"]):
            return True
            
        # Check for content overlap
        words = response_text.split()
        for i in range(len(words) - 1):
            phrase = ' '.join(words[i:i+2])
            if phrase in context_text:
                return True
                
        return False

    def _clean_response(self, text: str) -> str:
        """Clean and format the response."""
        # Remove any content before "Answer:"
        if "Answer:" in text:
            text = text.split("Answer:", 1)[1]
        
        # Clean up the text
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if line and not any(prefix in line.lower() for prefix in [
                "context:", "question:", "based on", "according to"
            ]):
                lines.append(line)
        
        return ' '.join(lines).strip()

    def _format_response(self, text: str) -> str:
        """Format an unformatted response according to our template."""
        # Split text into paragraphs
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        if not paragraphs:
            return "I apologize, but I couldn't generate a proper response."
        
        # Format the response
        formatted = []
        
        # Main concept (first paragraph)
        formatted.append("📌 Main Concept:")
        formatted.append(paragraphs[0])
        formatted.append("")
        
        # Key points (from subsequent paragraphs)
        formatted.append("📌 Key Points:")
        points = paragraphs[1:4] if len(paragraphs) > 1 else [paragraphs[0]]
        for point in points:
            formatted.append(f"• {point}")
        formatted.append("")
        
        # Additional information (if any remaining paragraphs)
        if len(paragraphs) > 4:
            formatted.append("📌 Additional Information:")
            formatted.append(paragraphs[4])
        
        return "\n".join(formatted)

    async def verify_document_storage(self, chatbot_id: str) -> dict:
        """Verify document storage and indexing for a chatbot."""
        try:
            chatbot = chatbots_collection.find_one({"_id": ObjectId(chatbot_id)})
            if not chatbot:
                raise HTTPException(status_code=404, detail="Chatbot not found")
            
            namespace = chatbot["pinecone_namespace"]
            index = self.pc.Index(INDEX_NAME)
            
            # Get stats for the namespace
            stats = index.describe_index_stats()
            namespace_stats = stats.namespaces.get(namespace, {})
            vector_count = namespace_stats.get('vector_count', 0)
            
            return {
                "documents": chatbot.get("documents", []),
                "vectors_stored": vector_count,
                "namespace": namespace,
                "status": "active" if vector_count > 0 else "empty"
            }
        except Exception as e:
            logger.error(f"Error verifying storage: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error verifying storage: {str(e)}") 