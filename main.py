import os
import json
import numpy as np
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import faiss
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="E-commerce FAQ RAG Chatbot", version="1.0.0")

# Initialize OpenAI client with your API key
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Global variables to store our data
vector_store = None
faq_chunks = []
embeddings_cache = []

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

class RAGChatbot:
    def __init__(self):
        self.embedding_model = "text-embedding-3-small"  # OpenAI embedding model
        self.chat_model = "gpt-4o-mini"  # OpenAI chat model (good for Arabic)
        self.chunk_size = 500
        self.overlap = 50
        
    def load_markdown_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load and parse your FAQ markdown file into chunks."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Split content into Q&A pairs based on ## headers
            chunks = []
            sections = content.split('\n##')  # Split by ## which marks questions
            
            for i, section in enumerate(sections):
                if section.strip():
                    # Clean up section
                    if not section.startswith('##'):
                        section = '##' + section
                    
                    lines = section.strip().split('\n')
                    if len(lines) >= 2:
                        question = lines[0].replace('#', '').strip()
                        answer = '\n'.join(lines[1:]).strip()
                        
                        chunks.append({
                            'id': f"faq_{i}",
                            'question': question,
                            'answer': answer,
                            'content': f"السؤال: {question}\nالجواب: {answer}",
                            'metadata': {'type': 'faq', 'index': i}
                        })
            
            return chunks
            
        except Exception as e:
            raise Exception(f"Error loading markdown file: {str(e)}")
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using OpenAI API."""
        try:
            response = client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings, dtype=np.float32)
            
        except Exception as e:
            raise Exception(f"Error generating embeddings: {str(e)}")
    
    def create_vector_store(self, chunks: List[Dict[str, Any]]) -> faiss.Index:
        """Create FAISS vector store from FAQ chunks."""
        try:
            # Extract content for embedding
            texts = [chunk['content'] for chunk in chunks]
            
            # Generate embeddings using OpenAI
            embeddings = self.get_embeddings(texts)
            
            # Create FAISS index for similarity search
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            index.add(embeddings)
            
            return index, embeddings
            
        except Exception as e:
            raise Exception(f"Error creating vector store: {str(e)}")
    
    def search_similar(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar FAQ entries to the user's question."""
        try:
            # Generate embedding for user's question
            query_embedding = self.get_embeddings([query])
            faiss.normalize_L2(query_embedding)
            
            # Search in vector store
            scores, indices = vector_store.search(query_embedding, k)
            
            # Prepare results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1:  # Valid index
                    chunk = faq_chunks[idx]
                    results.append({
                        'content': chunk['content'],
                        'question': chunk['question'],
                        'answer': chunk['answer'],
                        'score': float(score),
                        'metadata': chunk['metadata']
                    })
            
            return results
            
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
            avg_score = np.mean([chunk['score'] for chunk in context_chunks]) if context_chunks else 0
            confidence = min(avg_score * 100, 95)  # Cap at 95%
            
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
rag_chatbot = RAGChatbot()

@app.on_event("startup")
async def load_faq_data():
    """This runs when the server starts - loads your FAQ data."""
    global vector_store, faq_chunks, embeddings_cache
    
    try:
        print("Loading FAQ data...")
        
        # Load FAQ chunks from your markdown file
        faq_chunks = rag_chatbot.load_markdown_file("faq.md")
        print(f"Loaded {len(faq_chunks)} FAQ entries")
        
        # Create vector store for similarity search
        vector_store, embeddings_cache = rag_chatbot.create_vector_store(faq_chunks)
        print("Vector store created successfully")
        
    except Exception as e:
        print(f"Error loading FAQ data: {str(e)}")
        raise e

@app.get("/")
async def root():
    """Health check endpoint - test if server is running."""
    return {
        "message": "E-commerce FAQ RAG Chatbot API",
        "status": "running",
        "faq_entries": len(faq_chunks) if faq_chunks else 0
    }

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Main endpoint - this is what your frontend will call."""
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        if not vector_store or not faq_chunks:
            raise HTTPException(status_code=503, detail="FAQ data not loaded")
        
        # Search for relevant FAQ entries
        similar_chunks = rag_chatbot.search_similar(
            request.question, 
            k=request.max_results
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
    global vector_store, faq_chunks, embeddings_cache
    
    try:
        faq_chunks = rag_chatbot.load_markdown_file("faq.md")
        vector_store, embeddings_cache = rag_chatbot.create_vector_store(faq_chunks)
        
        return {
            "message": "FAQ data reloaded successfully",
            "entries_loaded": len(faq_chunks)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading FAQ: {str(e)}")

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)