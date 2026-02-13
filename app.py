"""
FastAPI application for TDI Chatbot RAG system
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn
import json
import logging

from rag_pipeline import get_pipeline
from config import config

from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize pipeline on startup using lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG pipeline on startup"""
    try:
        logger.info("Initializing RAG pipeline...")
        get_pipeline()
        logger.info("RAG pipeline initialized successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        raise

# Create FastAPI app
app = FastAPI(
    title="TDI Chatbot API",
    description="RAG-based chatbot API for The Data Island company information",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class ChatRequest(BaseModel):
    """Chat request model"""
    message: str = Field(..., description="User's message/question")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID for tracking")
    stream: bool = Field(False, description="Whether to stream the response")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "What services does TDI offer?",
                "conversation_id": "session-123",
                "stream": False
            }
        }
    }


class Source(BaseModel):
    """Source document model"""
    chunk_id: Any
    text: str
    distance: float
    source: str


class ChatResponse(BaseModel):
    """Chat response model"""
    response: str
    sources: List[Source]
    conversation_id: Optional[str] = None
    model: str
    guardrails_passed: bool
    error: Optional[str] = None
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "response": "The Data Island (TDI) offers AI and Data innovation services...",
                "sources": [
                    {
                        "chunk_id": 3,
                        "text": "Services include...",
                        "distance": 1.23,
                        "source": "TDI Profile.pdf"
                    }
                ],
                "conversation_id": "session-123",
                "model": "openai/gpt-oss-20b",
                "guardrails_passed": True,
                "error": None
            }
        }
    }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    model: str
    collection: str
    document_count: int


# Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "TDI Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        pipeline = get_pipeline()
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            model=config.MODEL_NAME,
            collection=config.COLLECTION_NAME,
            document_count=pipeline.collection.count()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    
    Process user's question using RAG pipeline and return response.
    """
    try:
        logger.info(f"Received chat request: {request.message[:100]}...")
        
        # Get pipeline
        pipeline = get_pipeline()
        
        # Check if streaming is requested
        if request.stream:
            # Return streaming response
            async def generate():
                for chunk in pipeline.stream_response(request.message):
                    if 'error' in chunk:
                        yield f"data: {json.dumps({'error': chunk['error']})}\n\n"
                        break
                    elif 'chunk' in chunk:
                        yield f"data: {json.dumps({'chunk': chunk['chunk']})}\n\n"
                    elif 'done' in chunk:
                        yield f"data: {json.dumps({'sources': chunk['sources'], 'done': True})}\n\n"
            
            return StreamingResponse(
                generate(),
                media_type="text/event-stream"
            )
        
        # Non-streaming response
        result = pipeline.query(request.message)
        
        # Check for errors
        if not result.get('guardrails_passed', False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get('error', 'Invalid request')
            )
        
        # Format response
        response = ChatResponse(
            response=result['response'],
            sources=[Source(**source) for source in result['sources']],
            conversation_id=request.conversation_id,
            model=result['model'],
            guardrails_passed=result['guardrails_passed'],
            error=result.get('error')
        )
        
        logger.info(f"Successfully generated response for query")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint.
    
    Stream response chunks as they are generated.
    """
    try:
        logger.info(f"Received streaming chat request: {request.message[:100]}...")
        
        # Get pipeline
        pipeline = get_pipeline()
        
        # Stream response
        async def generate():
            full_response = ""
            sources = []
            
            for chunk in pipeline.stream_response(request.message):
                if 'error' in chunk:
                    yield f"data: {json.dumps({'error': chunk['error']})}\n\n"
                    break
                elif 'chunk' in chunk:
                    full_response += chunk['chunk']
                    yield f"data: {json.dumps({'chunk': chunk['chunk']})}\n\n"
                elif 'done' in chunk:
                    sources = chunk.get('sources', [])
                    yield f"data: {json.dumps({'done': True, 'sources': sources, 'full_response': full_response})}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        logger.error(f"Error processing streaming chat request: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
