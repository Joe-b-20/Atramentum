# scripts/serve_api.py
#!/usr/bin/env python3
"""
FastAPI server with Basic Auth. Your despair, served fresh.
"""

import os
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import secrets
import uvicorn

from scripts.generate_with_rag import RAGGenerator
from src.atramentum.utils import logging as alog


# Initialize app
app = FastAPI(
    title="Atramentum API",
    description="Self-hosted journal generation with existential dread",
    version="0.1.2"
)

# Security
security = HTTPBasic()

# Logger
logger = alog.get_logger(__name__)

# Global generator (initialized on startup)
generator = None


class GenerateRequest(BaseModel):
    """Request model for generation."""
    prompt: str
    mode: str = "generate"  # generate, seed, rewrite
    k: int = 4
    max_new_tokens: int = 800
    temperature: float = 0.8
    top_p: float = 0.95
    use_rag: bool = True


class GenerateResponse(BaseModel):
    """Response model for generation."""
    text: str
    memories_used: int


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """Verify HTTP Basic Auth credentials."""
    correct_username = os.environ.get("BASIC_AUTH_USER")
    correct_password = os.environ.get("BASIC_AUTH_PASS")
    
    if not correct_username or not correct_password:
        logger.error("Basic auth credentials not set in environment")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server authentication not configured"
        )
    
    is_valid = (
        secrets.compare_digest(credentials.username, correct_username) and
        secrets.compare_digest(credentials.password, correct_password)
    )
    
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    return credentials.username


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    global generator
    
    logger.info("Initializing Atramentum API...")
    
    # Load model
    model_name = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    adapter_path = os.environ.get("ADAPTER_PATH", "checkpoints/atramentum-llama3-sft")
    index_dir = os.environ.get("INDEX_DIR", "index/faiss/bge_small")
    
    if not Path(adapter_path).exists():
        logger.warning(f"Adapter not found at {adapter_path}, using base model")
        adapter_path = None
    
    generator = RAGGenerator(model_name, adapter_path, index_dir)
    logger.info("Model loaded. Ready to channel despair.")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "message": "Still functioning, somehow"}


@app.post("/generate", response_model=GenerateResponse)
async def generate(
    request: GenerateRequest,
    username: str = Depends(verify_credentials)
):
    """Generate a journal entry with optional RAG."""
    logger.info(f"Generation request from {username}: mode={request.mode}, rag={request.use_rag}")
    
    try:
        # Retrieve memories if RAG enabled
        memories = ""
        if request.use_rag:
            memories = generator.retrieve_memories(request.prompt, request.k)
        
        # Generate text
        result = generator.generator.generate(
            prompt=request.prompt,
            memory=memories,
            mode=request.mode,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        # Ensure date format
        if not result.strip().startswith(datetime.now().strftime('%m/%d/%Y')):
            today = datetime.now().strftime('%m/%d/%Y')
            result = f"{today} —\n{result}"
        
        return GenerateResponse(
            text=result,
            memories_used=request.k if request.use_rag else 0
        )
    
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Generation failed. The void gazed back."
        )


# CORS configuration
if os.environ.get("CORS_LOCALHOST_ONLY", "1") == "1":
    origins = [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ]
else:
    origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    
    # Check auth is configured
    if not os.environ.get("BASIC_AUTH_USER") or not os.environ.get("BASIC_AUTH_PASS"):
        print("ERROR: Set BASIC_AUTH_USER and BASIC_AUTH_PASS environment variables")
        sys.exit(1)
    
    # Run server
    uvicorn.run(
        "serve_api:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )