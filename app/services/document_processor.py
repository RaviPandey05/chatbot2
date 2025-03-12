import io
import PyPDF2
from typing import List, Dict
import httpx
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import uuid
from app.config import (
    PINECONE_API_KEY, 
    EMBEDDING_MODEL,
    INDEX_NAME,
    PINECONE_ENVIRONMENT
)
from app.database import documents_collection
from datetime import datetime
import logging
import pandas as pd
import hashlib
import asyncio
from urllib.parse import urlparse
from collections import Counter
from PyPDF2 import PdfReader
from docx import Document
from fastapi import HTTPException
import re
import magic
import os
from bson import ObjectId
from app.database import chatbots_collection

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.SUPPORTED_MIME_TYPES = {
            'application/pdf': 'pdf',
            'text/plain': 'txt',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx'
        }
        
    def extract_text(self, content: bytes, mime_type: str) -> str:
        file_type = self.SUPPORTED_MIME_TYPES.get(mime_type)
        if not file_type:
            raise HTTPException(400, "Unsupported file type")
        
        try:
            if file_type == "pdf":
                return self._extract_text_from_pdf(content)
            elif file_type == "docx":
                return self._extract_text_from_docx(content)
            elif file_type == "excel":
                return self._extract_text_from_excel(content)
            elif file_type == "csv":
                return self._extract_text_from_csv(content)
            elif file_type == "text":
                return self._extract_text_from_txt(content)
        except Exception as e:
            logger.error(f"Text extraction failed: {str(e)}")
            raise HTTPException(400, f"Error processing {file_type} file")

    async def process_file(self, file_content: bytes, filename: str, mime_type: str, chatbot_id: str, namespace: str) -> dict:
        try:
            logger.info(f"Starting to process file: {filename} for chatbot: {chatbot_id}")
            
            # Update initial status
            chatbots_collection.update_one(
                {"_id": ObjectId(chatbot_id)},
                {"$set": {"processing_status": "extracting_text"}}
            )

            # Extract text
            text = await self._extract_text(file_content, mime_type)
            if not text:
                raise ValueError("No text could be extracted from the file")

            # Split into chunks with progress tracking
            chunks = self._split_text(text, chunk_size=1000, overlap=100)
            if not chunks:
                raise ValueError("No valid text chunks created")

            logger.info(f"Created {len(chunks)} chunks from {filename}")
            total_chunks = len(chunks)
            vectors = []

            # Process chunks in smaller batches
            batch_size = 10
            for i in range(0, total_chunks, batch_size):
                batch_chunks = chunks[i:i + batch_size]
                
                # Update progress
                progress = min((i + batch_size) / total_chunks * 100, 100)
                chatbots_collection.update_one(
                    {"_id": ObjectId(chatbot_id)},
                    {"$set": {
                        "processing_status": "creating_embeddings",
                        "progress": progress
                    }}
                )

                # Create embeddings for batch
                embeddings = self.embedding_model.encode(
                    batch_chunks,
                    show_progress_bar=False,
                    batch_size=5
                )

                # Create vectors for the batch
                for j, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings)):
                    chunk_id = f"{filename}_{i+j}"
                    vectors.append({
                        "id": chunk_id,
                        "values": embedding.tolist(),
                        "metadata": {
                            "text": chunk,
                            "source": filename,
                            "chunk_id": chunk_id,
                            "position": i+j
                        }
                    })

            # Update status for vector upload
            chatbots_collection.update_one(
                {"_id": ObjectId(chatbot_id)},
                {"$set": {"processing_status": "uploading_vectors"}}
            )

            # Upload vectors in smaller batches
            index = self.pc.Index(INDEX_NAME)
            upload_batch_size = 20
            for i in range(0, len(vectors), upload_batch_size):
                batch = vectors[i:i + upload_batch_size]
                index.upsert(vectors=batch, namespace=namespace)
                logger.info(f"Uploaded batch {i//upload_batch_size + 1} of vectors")

            # Update final status
            chatbots_collection.update_one(
                {"_id": ObjectId(chatbot_id)},
                {
                    "$addToSet": {"documents": filename},
                    "$set": {
                        "last_updated": datetime.utcnow(),
                        "processing_status": "completed",
                        "chunks_processed": len(chunks),
                        "vectors_stored": len(vectors),
                        "progress": 100
                    }
                }
            )

            return {
                "status": "success",
                "message": f"Successfully processed {filename}",
                "chunks_processed": len(chunks),
                "vectors_uploaded": len(vectors)
            }

        except Exception as e:
            logger.error(f"Error processing file {filename}: {str(e)}")
            chatbots_collection.update_one(
                {"_id": ObjectId(chatbot_id)},
                {"$set": {
                    "processing_status": "failed",
                    "error_message": str(e)
                }}
            )
            raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

    async def process_url(self, url: str, chatbot_id: str, namespace: str):
        """Process webpage content and store in Pinecone with chatbot's namespace"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                html_text = response.text

            # Extract text from HTML
            soup = BeautifulSoup(html_text, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            
            if not text.strip():
                raise HTTPException(400, "No text content found at URL")

            # Create chunks and embeddings
            chunks = self._chunk_text_by_words(text)
            vectors = []
            
            for chunk in chunks:
                chunk_id = str(uuid.uuid4())
                embedding = self.embedding_model.encode(chunk).tolist()
                
                vector = {
                    "id": chunk_id,
                    "values": embedding,
                    "metadata": {
                        "text": chunk[:2000],
                        "source": url,
                        "type": "url",
                        "chatbot_id": chatbot_id,
                        "keywords": self._extract_keywords(chunk)
                    }
                }
                vectors.append(vector)

            # Store in Pinecone under chatbot's namespace
            index = self.pc.Index(INDEX_NAME)
            index.upsert(vectors=vectors, namespace=namespace)

            # Store URL metadata in MongoDB
            url_metadata = {
                "url": url,
                "chatbot_id": chatbot_id,
                "chunk_ids": [v["id"] for v in vectors],
                "processed_date": datetime.utcnow(),
                "chunk_count": len(chunks)
            }
            documents_collection.insert_one(url_metadata)

            return {"status": "success", "chunks_processed": len(chunks)}

        except Exception as e:
            logger.error(f"Error processing URL: {str(e)}")
            raise HTTPException(500, f"Error processing URL: {str(e)}")

    # Helper methods
    async def _process_pdf(self, content: bytes) -> str:
        """Process PDF file content and extract text efficiently."""
        try:
            pdf_file = io.BytesIO(content)
            pdf_reader = PdfReader(pdf_file)
            
            text_parts = []
            total_pages = len(pdf_reader.pages)
            
            # Process pages in smaller chunks
            chunk_size = 5
            for i in range(0, total_pages, chunk_size):
                chunk_pages = pdf_reader.pages[i:min(i + chunk_size, total_pages)]
                for page in chunk_pages:
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_parts.append(page_text.strip())
                    except Exception as e:
                        logger.warning(f"Error extracting text from page: {str(e)}")
                        continue
            
            if not text_parts:
                raise ValueError("No readable text found in PDF file")
            
            return "\n\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing PDF file: {str(e)}")

    async def _process_docx(self, content: bytes) -> str:
        """Process DOCX file content and extract text."""
        try:
            # Create document reader object
            doc_file = io.BytesIO(content)
            doc = Document(doc_file)
            
            # Extract text from paragraphs
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
            if not text.strip():
                raise HTTPException(
                    status_code=400,
                    detail="No readable text found in DOCX file"
                )
            
            return text.strip()
        except Exception as e:
            logger.error(f"Error processing DOCX: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing DOCX file: {str(e)}"
            )

    def _extract_text_from_pdf(self, content: bytes) -> str:
        """Legacy method - redirects to _process_pdf."""
        return asyncio.run(self._process_pdf(content))

    def _extract_text_from_docx(self, content: bytes) -> str:
        """Legacy method - redirects to _process_docx."""
        return asyncio.run(self._process_docx(content))

    def _extract_text_from_excel(self, content: bytes) -> str:
        df = pd.read_excel(io.BytesIO(content), engine="openpyxl")
        return df.to_csv(index=False)

    def _extract_text_from_csv(self, content: bytes) -> str:
        df = pd.read_csv(io.BytesIO(content))
        return df.to_csv(index=False)

    def _extract_text_from_txt(self, content: bytes) -> str:
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            return content.decode("latin-1")

    def _chunk_text_by_chars(self, text: str, max_chars: int = 10000) -> List[str]:
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

    def _chunk_text_by_words(self, text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
        words = text.split()
        if not words:
            return []
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            i += chunk_size - overlap
        return chunks

    def _extract_keywords(self, text: str, num_keywords: int = 5) -> List[str]:
        words = re.findall(r'\w+', text.lower())
        stopwords = {"the", "and", "is", "in", "to", "of", "a", "for", "on", "with"}
        filtered_words = [w for w in words if w not in stopwords and len(w) > 3]
        most_common = Counter(filtered_words).most_common(num_keywords)
        return [word for word, count in most_common]

    async def _extract_text(self, file_content: bytes, mime_type: str) -> str:
        """Extract text from file content based on mime type."""
        try:
            if mime_type == 'application/pdf':
                return await self._process_pdf(file_content)
            elif mime_type == 'text/plain':
                return file_content.decode('utf-8')
            elif mime_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
                return await self._process_docx(file_content)
            else:
                raise ValueError(f"Unsupported mime type: {mime_type}")
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error extracting text: {str(e)}")

    def _split_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            
            # Adjust chunk to end at a sentence or paragraph break
            if end < text_length:
                for sep in ['\n\n', '\n', '. ', ' ']:
                    last_sep = chunk.rfind(sep)
                    if last_sep != -1:
                        end = start + last_sep + len(sep)
                        chunk = text[start:end]
                        break

            chunks.append(chunk.strip())
            start = end - overlap

        return [c for c in chunks if c]  # Remove empty chunks 