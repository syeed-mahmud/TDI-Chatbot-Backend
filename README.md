# TDI Chatbot - RAG System

A Retrieval-Augmented Generation (RAG) chatbot system for The Data Island (TDI) company information.

## Features

- 📚 **PDF Knowledge Base**: Ingests PDF documents into ChromaDB vector database
- 🤖 **LLM Integration**: Uses Groq's LLM (openai/gpt-oss-20b) for natural language responses
- 🔍 **Semantic Search**: Retrieves relevant context using MiniLM embeddings
- 🛡️ **Guardrails**: Input validation and output filtering for safety
- 🌊 **Streaming Support**: Real-time response streaming
- 🚀 **FastAPI**: REST API endpoints for easy integration

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment

Copy the example environment file and add your Groq API key:

```bash
copy .env.example .env
```

Edit `.env` and add your Groq API key:
```
GROQ_API_KEY=your_actual_groq_api_key
```

### 3. Ingest PDF (First Time Only)

If you haven't ingested the PDF yet:

```bash
python ingest_pdf_to_chroma.py
```

This will create a `chroma_db` directory with your vector database.

### 4. Start the API Server

```bash
python app.py
```

Or use uvicorn directly:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

## API Documentation

### Interactive Docs

Visit `http://localhost:8000/docs` for interactive Swagger UI documentation.

### Endpoints

#### 1. Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model": "openai/gpt-oss-20b",
  "collection": "tdi_documents",
  "document_count": 21
}
```

#### 2. Chat (Non-Streaming)
```bash
POST /chat
Content-Type: application/json

{
  "message": "What services does TDI offer?",
  "conversation_id": "optional-session-id",
  "stream": false
}
```

**Response:**
```json
{
  "response": "The Data Island offers...",
  "sources": [
    {
      "chunk_id": 3,
      "text": "...",
      "distance": 1.23,
      "source": "TDI Profile.pdf"
    }
  ],
  "conversation_id": "optional-session-id",
  "model": "openai/gpt-oss-20b",
  "guardrails_passed": true,
  "error": null
}
```

#### 3. Chat (Streaming)
```bash
POST /chat/stream
Content-Type: application/json

{
  "message": "Tell me about TDI's mission"
}
```

Returns Server-Sent Events (SSE) with response chunks.

## Example Usage

### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Chat request
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is TDI?",
    "stream": false
  }'
```

### Using Python

```python
import requests

# Chat request
response = requests.post(
    "http://localhost:8000/chat",
    json={
        "message": "What services does TDI offer?",
        "stream": False
    }
)

result = response.json()
print(result['response'])
```

### Using JavaScript/Fetch

```javascript
// Non-streaming
fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    message: 'What is TDI?',
    stream: false
  })
})
.then(response => response.json())
.then(data => console.log(data.response));

// Streaming
const eventSource = new EventSource('http://localhost:8000/chat/stream');
eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.chunk) {
    console.log(data.chunk);
  }
};
```

## Configuration

All configuration is done through environment variables in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Your Groq API key | Required |
| `MODEL_NAME` | Groq model to use | openai/gpt-oss-20b |
| `TEMPERATURE` | LLM temperature (0-1) | 0.7 |
| `MAX_TOKENS` | Max response tokens | 8192 |
| `TOP_K_RESULTS` | Number of context chunks | 3 |
| `CHROMA_DB_PATH` | Vector DB path | ./chroma_db |

## Project Structure

```
TDI Chatbot/
├── app.py                      # FastAPI application
├── rag_pipeline.py            # RAG pipeline with LangChain
├── config.py                  # Configuration management
├── guardrails.py              # Input/output validation
├── ingest_pdf_to_chroma.py   # PDF ingestion script
├── requirements.txt           # Python dependencies
├── .env.example              # Environment template
├── .env                      # Your environment (create this)
├── chroma_db/                # Vector database (created after ingestion)
└── README.md                 # This file
```

## How It Works

1. **Document Ingestion**: PDF is split into chunks and embedded using MiniLM
2. **Query Processing**: User question is embedded and similar chunks are retrieved
3. **Context Building**: Top-K most relevant chunks are combined as context
4. **LLM Generation**: Groq LLM generates response using context + question
5. **Guardrails**: Input/output validation ensures safe responses
6. **API Response**: FastAPI returns JSON with answer and sources

## Troubleshooting

**Issue**: `GROQ_API_KEY environment variable is required`
- **Solution**: Make sure you created `.env` file and added your API key

**Issue**: `ChromaDB path does not exist`
- **Solution**: Run `python ingest_pdf_to_chroma.py` first to create the database

**Issue**: NumPy compatibility error
- **Solution**: Make sure numpy<2.0 is installed: `pip install "numpy<2.0"`

## Development

To run in development mode with auto-reload:

```bash
uvicorn app:app --reload
```

## License

This project is for The Data Island (TDI) internal use.
