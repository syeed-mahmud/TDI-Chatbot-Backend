"""
PDF Ingestion Script for ChromaDB
This script ingests a PDF document into ChromaDB vector database using MiniLM embeddings.
"""

import os
from pathlib import Path
from typing import List

# PDF processing
import PyPDF2

# ChromaDB for vector storage
import chromadb
from chromadb.config import Settings

# Sentence transformers for embeddings
from sentence_transformers import SentenceTransformer


class PDFToChromaIngester:
    def __init__(self, 
                 pdf_path: str, 
                 collection_name: str = "tdi_documents",
                 persist_directory: str = "./chroma_db",
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the PDF ingestion system.
        
        Args:
            pdf_path: Path to the PDF file
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist ChromaDB
            model_name: SentenceTransformer model name for embeddings
        """
        self.pdf_path = pdf_path
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize embedding model (MiniLM)
        print(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        
        # Initialize ChromaDB PersistentClient (to actually save to disk)
        print(f"Initializing ChromaDB at: {persist_directory}")
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "TDI Profile documents"}
        )
    
    def extract_text_from_pdf(self) -> str:
        """Extract text from PDF file."""
        print(f"Extracting text from: {self.pdf_path}")
        
        text = ""
        with open(self.pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            print(f"Total pages: {num_pages}")
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text()
                text += page_text
                print(f"Processed page {page_num}/{num_pages}")
        
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[dict]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: Input text to chunk
            chunk_size: Number of characters per chunk
            overlap: Number of overlapping characters between chunks
            
        Returns:
            List of dictionaries containing chunks and metadata
        """
        print(f"Chunking text (chunk_size={chunk_size}, overlap={overlap})")
        
        chunks = []
        start = 0
        text_length = len(text)
        chunk_id = 0
        
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            
            # Clean up the chunk
            chunk = chunk.strip()
            
            if chunk:  # Only add non-empty chunks
                chunks.append({
                    'id': f"chunk_{chunk_id}",
                    'text': chunk,
                    'metadata': {
                        'source': os.path.basename(self.pdf_path),
                        'chunk_id': chunk_id,
                        'start_char': start,
                        'end_char': end
                    }
                })
                chunk_id += 1
            
            start += (chunk_size - overlap)
        
        print(f"Created {len(chunks)} chunks")
        return chunks
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using MiniLM model."""
        print("Generating embeddings...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()
    
    def ingest_to_chromadb(self, chunks: List[dict]):
        """
        Ingest chunks into ChromaDB.
        
        Args:
            chunks: List of text chunks with metadata
        """
        print("Ingesting into ChromaDB...")
        
        # Prepare data for ChromaDB
        ids = [chunk['id'] for chunk in chunks]
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        
        print(f"Successfully ingested {len(chunks)} chunks into ChromaDB")
        print(f"Collection '{self.collection_name}' now contains {self.collection.count()} documents")
    
    def run(self):
        """Execute the full ingestion pipeline."""
        print("=" * 50)
        print("Starting PDF Ingestion Pipeline")
        print("=" * 50)
        
        # Step 1: Extract text from PDF
        text = self.extract_text_from_pdf()
        print(f"\nExtracted {len(text)} characters")
        
        # Step 2: Chunk the text
        chunks = self.chunk_text(text)
        
        # Step 3: Ingest into ChromaDB (embeddings generated internally)
        self.ingest_to_chromadb(chunks)
        
        print("\n" + "=" * 50)
        print("Ingestion Complete!")
        print("=" * 50)
        
        return self.collection


def main():
    """Main execution function."""
    # Configuration
    PDF_PATH = r"d:/TDI Project/TDI Chatbot/TDI Business Profile.pdf"
    COLLECTION_NAME = "tdi_documents"
    PERSIST_DIRECTORY = "./chroma_db"
    
    # Verify PDF exists
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF file not found: {PDF_PATH}")
    
    # Create ingester and run
    ingester = PDFToChromaIngester(
        pdf_path=PDF_PATH,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIRECTORY
    )
    
    collection = ingester.run()
    
    # Example: Query the collection
    print("\n" + "=" * 50)
    print("Testing Query")
    print("=" * 50)
    
    query_text = "What is TDI?"
    query_embedding = ingester.embedding_model.encode([query_text]).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )
    
    print(f"\nQuery: '{query_text}'")
    print("\nTop 3 Results:")
    for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0]), 1):
        print(f"\n{i}. (Distance: {distance:.4f})")
        print(f"   {doc[:200]}...")


if __name__ == "__main__":
    main()
