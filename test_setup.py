"""
Simple test script to verify RAG pipeline setup
Run this after setting up your .env file
"""
import os
from pathlib import Path

def test_imports():
    """Test if all required packages are importable"""
    print("Testing imports...")
    try:
        import chromadb
        print("✓ chromadb")
        
        import langchain
        print("✓ langchain")
        
        import langchain_groq
        print("✓ langchain-groq")
        
        from groq import Groq
        print("✓ groq")
        
        import fastapi
        print("✓ fastapi")
        
        from sentence_transformers import SentenceTransformer
        print("✓ sentence-transformers")
        
        print("\n✅ All imports successful!")
        return True
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        return False


def test_environment():
    """Test if environment variables are set"""
    print("\nTesting environment variables...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key and groq_key != "your_groq_api_key_here":
        print("✓ GROQ_API_KEY is set")
    else:
        print("❌ GROQ_API_KEY is not set or using example value")
        print("   Please update your .env file with a valid Groq API key")
        return False
    
    print("\n✅ Environment configured!")
    return True


def test_chromadb():
    """Test if ChromaDB exists and has data"""
    print("\nTesting ChromaDB...")
    
    chroma_path = Path("./chroma_db")
    if not chroma_path.exists():
        print("❌ ChromaDB directory not found")
        print("   Please run: python ingest_pdf_to_chroma.py")
        return False
    
    try:
        import chromadb
        from chromadb.config import Settings
        
        client = chromadb.Client(Settings(
            persist_directory="./chroma_db",
            anonymized_telemetry=False
        ))
        
        collection = client.get_collection("tdi_documents")
        doc_count = collection.count()
        
        print(f"✓ ChromaDB found with {doc_count} documents")
        print("\n✅ ChromaDB ready!")
        return True
    except Exception as e:
        print(f"❌ ChromaDB test failed: {e}")
        return False


def test_rag_pipeline():
    """Test if RAG pipeline initializes"""
    print("\nTesting RAG pipeline initialization...")
    
    try:
        from rag_pipeline import get_pipeline
        
        pipeline = get_pipeline()
        print("✓ RAG pipeline initialized")
        
        # Test a simple query
        result = pipeline.query("What is TDI?")
        
        if result['guardrails_passed']:
            print("✓ Query test passed")
            print(f"\nSample response: {result['response'][:100]}...")
            print("\n✅ RAG pipeline working!")
            return True
        else:
            print("❌ Query test failed guardrails")
            return False
            
    except Exception as e:
        print(f"❌ RAG pipeline test failed: {e}")
        print("\nMake sure:")
        print("  1. .env file has valid GROQ_API_KEY")
        print("  2. ChromaDB exists (run ingest_pdf_to_chroma.py)")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("TDI Chatbot - System Test")
    print("=" * 60)
    
    results = []
    
    # Test imports
    results.append(("Imports", test_imports()))
    
    # Test environment
    results.append(("Environment", test_environment()))
    
    # Test ChromaDB
    results.append(("ChromaDB", test_chromadb()))
    
    # Test RAG pipeline
    results.append(("RAG Pipeline", test_rag_pipeline()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:20s}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 All tests passed! Your system is ready.")
        print("\nTo start the API server, run:")
        print("  python app.py")
        print("\nOr:")
        print("  uvicorn app:app --reload")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
    
    return all_passed


if __name__ == "__main__":
    main()
