#!/usr/bin/env python3
"""Simple API server for testing."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from datetime import datetime

app = FastAPI(title="Atramentum API")

class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 500

class GenerateResponse(BaseModel):
    text: str

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Server is running"}

@app.post("/generate")
async def generate(request: GenerateRequest):
    # For now, return a mock response
    today = datetime.now().strftime("%m/%d/%Y")
    mock_response = f"{today} —\nYour prompt was: {request.prompt}\n\n(Model generation will be here once trained)"
    
    return GenerateResponse(text=mock_response)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on port {port}")
    print(f"API docs at http://localhost:{port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=port)
