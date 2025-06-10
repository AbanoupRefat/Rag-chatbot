import os
import json
import uuid
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="E-commerce FAQ RAG Chatbot with ChromaDB", version="1.0.0")

# Initialize OpenAI client with your API key
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Global variables to store our data
chroma_client = None
collection = None
faq_chunks = []

# Define the request format that frontend will send
class QueryRequest(BaseModel):
    question: str
    max_results: int = 3
    language: str = "ar"  # Arabic by default

# Define the response format that we'll send back
class QueryResponse(BaseModel):
    answer: str
    relevant_sources: List[str]
    confidence: float
    status: str

class ChromaRAGChatbot:
    def __init__(self):
        self.chat_model = "gpt-4o-mini"  # OpenAI chat model (good for Arabic)
        self.collection_name = "faq_collection"
        
    def initialize_chromadb(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Initialize ChromaDB client (persistent storage)
            self.chroma_client = chromadb.PersistentClient(
                path="./chroma_db",  # This will create a local database
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Create or get collection with OpenAI embeddings
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model_name="text-embedding-3-small"
                ),
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            
            return self.chroma_client, self.collection
            
        except Exception as e:
            raise Exception(f"Error initializing ChromaDB: {str(e)}")
    
    def load_markdown_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load and parse your FAQ markdown file into chunks."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Split content into Q&A pairs based on ### headers
            chunks = []
            sections = content.split('\n###')  # Split by ### which marks questions
            
            for i, section in enumerate(sections):
                if section.strip():
                    # Clean up section
                    if not section.startswith('###'):
                        section = '###' + section
                    
                    lines = section.strip().split('\n')
                    if len(lines) >= 2:
                        question = lines[0].replace('#', '').strip()
                        answer = '\n'.join(lines[1:]).strip()
                        
                        chunks.append({
                            'id': f"faq_{i}",
                            'question': question,
                            'answer': answer,
                            'content': f"السؤال: {question}\nالجواب: {answer}",
                            'metadata': {'type': 'faq', 'index': i, 'question': question}
                        })
            
            return chunks
            
        except Exception as e:
            raise Exception(f"Error loading markdown file: {str(e)}")
    
    def add_documents_to_chroma(self, chunks: List[Dict[str, Any]]):
        """Add FAQ chunks to ChromaDB collection."""
        try:
            # Check if collection already has documents
            collection_count = self.collection.count()
            if collection_count > 0:
                print(f"Collection already contains {collection_count} documents. Skipping...")
                return
            
            # Prepare data for ChromaDB
            documents = []
            metadatas = []
            ids = []
            
            for chunk in chunks:
                documents.append(chunk['content'])
                metadatas.append({
                    'question': chunk['question'],
                    'answer': chunk['answer'],
                    'type': chunk['metadata']['type'],
                    'index': chunk['metadata']['index']
                })
                ids.append(chunk['id'])
            
            # Add documents to collection (ChromaDB will handle embeddings automatically)
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"Added {len(documents)} documents to ChromaDB collection")
            
        except Exception as e:
            raise Exception(f"Error adding documents to ChromaDB: {str(e)}")
    
    def search_similar(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Search for similar FAQ entries using ChromaDB."""
        try:
            # Query the collection
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Process results
            similar_chunks = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity score (ChromaDB returns distances, lower = more similar)
                    similarity_score = max(0, 1 - distance)  # Convert distance to similarity
                    
                    similar_chunks.append({
                        'content': doc,
                        'question': metadata['question'],
                        'answer': metadata['answer'],
                        'score': similarity_score,
                        'metadata': metadata
                    })
            
            return similar_chunks
            
        except Exception as e:
            raise Exception(f"Error searching similar content: {str(e)}")
    
    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate response using OpenAI with the relevant FAQ context."""
        try:
            # Prepare context from similar FAQ entries
            context = "\n\n".join([chunk['content'] for chunk in context_chunks])
            
            # Create prompt in Arabic
            system_prompt = """أنت مساعد ذكي متخصص في الإجابة على أسئلة العملاء حول منصة مزادات التجارة الإلكترونية.
            
قواعد مهمة:
1. استخدم فقط المعلومات المتوفرة في السياق المعطى
2. أجب باللغة العربية بوضوح ومهنية
3. إذا لم تجد إجابة في السياق، قل "عذراً، لا أملك معلومات كافية للإجابة على هذا السؤال"
4. كن مختصراً ومفيداً
5. استخدم نبرة ودودة ومهنية"""

            user_prompt = f"""السياق المتاح:
{context}

السؤال: {query}

الرجاء الإجابة بناءً على السياق المتوفر فقط."""

            # Generate response using OpenAI
            response = client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent answers
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            
            # Calculate confidence based on similarity scores
            if context_chunks:
                avg_score = sum(chunk['score'] for chunk in context_chunks) / len(context_chunks)
                confidence = min(avg_score * 100, 95)  # Cap at 95%
            else:
                confidence = 0
            
            return {
                'answer': answer,
                'relevant_sources': [chunk['question'] for chunk in context_chunks],
                'confidence': round(confidence, 2),
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'answer': 'عذراً، حدث خطأ في معالجة طلبك. يرجى المحاولة مرة أخرى.',
                'relevant_sources': [],
                'confidence': 0.0,
                'status': 'error',
                'error': str(e)
            }

# Initialize the chatbot
rag_chatbot = ChromaRAGChatbot()

@app.on_event("startup")
async def load_faq_data():
    """This runs when the server starts - loads your FAQ data."""
    global chroma_client, collection, faq_chunks
    
    try:
        print("Initializing ChromaDB...")
        
        # Initialize ChromaDB
        chroma_client, collection = rag_chatbot.initialize_chromadb()
        
        print("Loading FAQ data...")
        
        # Load FAQ chunks from your markdown file
        faq_chunks = rag_chatbot.load_markdown_file("faq.md")
        print(f"Loaded {len(faq_chunks)} FAQ entries")
        
        # Add documents to ChromaDB collection
        rag_chatbot.add_documents_to_chroma(faq_chunks)
        
        print("ChromaDB setup completed successfully")
        
    except Exception as e:
        print(f"Error loading FAQ data: {str(e)}")
        raise e

@app.get("/")
async def root():
    """Health check endpoint - test if server is running."""
    collection_count = collection.count() if collection else 0
    return {
        "message": "E-commerce FAQ RAG Chatbot API with ChromaDB",
        "status": "running",
        "faq_entries": len(faq_chunks) if faq_chunks else 0,
        "chromadb_documents": collection_count
    }

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Main endpoint - this is what your frontend will call."""
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        if not collection or not faq_chunks:
            raise HTTPException(status_code=503, detail="FAQ data not loaded")
        
        # Search for relevant FAQ entries using ChromaDB
        similar_chunks = rag_chatbot.search_similar(
            request.question, 
            n_results=request.max_results
        )
        
        if not similar_chunks:
            return QueryResponse(
                answer="عذراً، لم أجد معلومات ذات صلة بسؤالك في قاعدة البيانات المتاحة.",
                relevant_sources=[],
                confidence=0.0,
                status="no_results"
            )
        
        # Generate response using OpenAI
        result = rag_chatbot.generate_response(request.question, similar_chunks)
        
        return QueryResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/faq/list")
async def list_faq():
    """List all FAQ entries - useful for debugging."""
    if not faq_chunks:
        raise HTTPException(status_code=503, detail="FAQ data not loaded")
    
    return {
        "total": len(faq_chunks),
        "chromadb_count": collection.count() if collection else 0,
        "entries": [
            {
                "id": chunk["id"],
                "question": chunk["question"],
                "preview": chunk["answer"][:100] + "..." if len(chunk["answer"]) > 100 else chunk["answer"]
            }
            for chunk in faq_chunks
        ]
    }

@app.post("/faq/reload")
async def reload_faq():
    """Reload FAQ data without restarting server."""
    global faq_chunks
    
    try:
        # Clear existing collection
        if collection:
            collection.delete()
        
        # Reinitialize
        chroma_client, collection = rag_chatbot.initialize_chromadb()
        
        # Reload FAQ data
        faq_chunks = rag_chatbot.load_markdown_file("faq.md")
        rag_chatbot.add_documents_to_chroma(faq_chunks)
        
        return {
            "message": "FAQ data reloaded successfully",
            "entries_loaded": len(faq_chunks),
            "chromadb_documents": collection.count()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading FAQ: {str(e)}")

@app.get("/chromadb/stats")
async def chromadb_stats():
    """Get ChromaDB collection statistics."""
    if not collection:
        raise HTTPException(status_code=503, detail="ChromaDB not initialized")
    
    try:
        # Get collection info
        collection_count = collection.count()
        
        # Get sample documents
        sample_results = collection.peek(limit=3)
        
        return {
            "collection_name": rag_chatbot.collection_name,
            "document_count": collection_count,
            "sample_documents": sample_results.get('documents', [])[:3] if sample_results else [],
            "embedding_function": "OpenAI text-embedding-3-small"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting ChromaDB stats: {str(e)}")

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
