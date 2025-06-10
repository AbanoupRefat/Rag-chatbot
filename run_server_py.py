#!/usr/bin/env python3
"""
Simple script to run the RAG chatbot server.
"""

import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Start the FastAPI server."""
    print("🚀 Starting RAG Chatbot Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📚 API Documentation will be at: http://localhost:8000/docs")
    print("⛔ Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Check if OpenAI API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY not found in .env file!")
        print("Please create a .env file with your OpenAI API key.")
        return
    
    # Check if FAQ file exists
    if not os.path.exists("faq.md"):
        print("❌ ERROR: faq.md file not found!")
        print("Please create your faq.md file in the same directory.")
        return
    
    print("✅ Configuration looks good!")
    print("🔄 Loading FAQ data and starting server...")
    
    # Start the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )

if __name__ == "__main__":
    main()