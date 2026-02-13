"""
RAG Pipeline using LangChain, ChromaDB, and Groq
"""
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from config import config
from guardrails import guardrails


class RAGPipeline:
    """RAG pipeline for TDI chatbot"""
    
    def __init__(self):
        """Initialize the RAG pipeline"""
        # Validate configuration
        config.validate()
        
        # Initialize embedding model
        print(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        
        # Initialize ChromaDB PersistentClient
        print(f"Connecting to ChromaDB at: {config.CHROMA_DB_PATH}")
        self.chroma_client = chromadb.PersistentClient(
            path=config.CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get collection
        self.collection = self.chroma_client.get_collection(
            name=config.COLLECTION_NAME
        )
        print(f"Connected to collection '{config.COLLECTION_NAME}' with {self.collection.count()} documents")
        
        # Initialize Groq LLM
        print(f"Initializing Groq LLM: {config.MODEL_NAME}")
        self.llm = ChatGroq(
            groq_api_key=config.GROQ_API_KEY,
            model_name=config.MODEL_NAME,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
        )
        
        # Create prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=config.SYSTEM_PROMPT
        )
        
        print("RAG Pipeline initialized successfully!")
    
    def retrieve_context(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from ChromaDB.
        
        Args:
            query: User's query
            top_k: Number of results to retrieve (default from config)
            
        Returns:
            List of relevant documents with metadata
        """
        if top_k is None:
            top_k = config.TOP_K_RESULTS
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        # Format results
        sources = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            sources.append({
                'chunk_id': metadata.get('chunk_id', i),
                'text': doc,
                'distance': float(distance),
                'source': metadata.get('source', 'unknown')
            })
        
        return sources
    
    def generate_response(
        self,
        query: str,
        context: str,
        stream: bool = False
    ) -> str:
        """
        Generate response using Groq LLM.
        
        Args:
            query: User's question
            context: Retrieved context
            stream: Whether to stream the response
            
        Returns:
            Generated response
        """
        # Format prompt
        prompt = self.prompt_template.format(
            context=context,
            question=query
        )
        
        # Generate response
        if stream:
            # Streaming is handled separately in the API
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        else:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
    
    def query(
        self,
        user_query: str,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve context and generate response.
        
        Args:
            user_query: User's question
            stream: Whether to stream the response
            
        Returns:
            Dictionary with response, sources, and metadata
        """
        # Sanitize input
        sanitized_query = guardrails.sanitize_input(user_query)
        
        # Check input guardrails
        is_valid, error_msg = guardrails.check_input(sanitized_query)
        if not is_valid:
            return {
                'response': error_msg,
                'sources': [],
                'guardrails_passed': False,
                'error': error_msg
            }
        
        # Retrieve context
        sources = self.retrieve_context(sanitized_query)
        
        # Combine context
        context = "\n\n".join([
            f"[Source {i+1}]: {source['text']}"
            for i, source in enumerate(sources)
        ])
        
        # Generate response
        response = self.generate_response(
            query=sanitized_query,
            context=context,
            stream=stream
        )
        
        # Check output guardrails
        is_valid, error_msg = guardrails.check_output(response, context)
        if not is_valid:
            return {
                'response': error_msg,
                'sources': sources,
                'guardrails_passed': False,
                'error': error_msg
            }
        
        return {
            'response': response,
            'sources': sources,
            'guardrails_passed': True,
            'model': config.MODEL_NAME,
            'context_used': context
        }
    
    def stream_response(self, user_query: str):
        """
        Stream response from Groq LLM.
        
        Args:
            user_query: User's question
            
        Yields:
            Response chunks
        """
        # Sanitize input
        sanitized_query = guardrails.sanitize_input(user_query)
        
        # Check input guardrails
        is_valid, error_msg = guardrails.check_input(sanitized_query)
        if not is_valid:
            yield {'error': error_msg}
            return
        
        # Retrieve context
        sources = self.retrieve_context(sanitized_query)
        
        # Combine context
        context = "\n\n".join([
            f"[Source {i+1}]: {source['text']}"
            for i, source in enumerate(sources)
        ])
        
        # Format prompt
        prompt = self.prompt_template.format(
            context=context,
            question=sanitized_query
        )
        
        # Stream response
        for chunk in self.llm.stream([HumanMessage(content=prompt)]):
            if chunk.content:
                yield {'chunk': chunk.content}
        
        # Send sources at the end
        yield {'sources': sources, 'done': True}


# Create singleton instance (will be initialized when imported)
_pipeline: Optional[RAGPipeline] = None

def get_pipeline() -> RAGPipeline:
    """Get or create RAG pipeline instance"""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
