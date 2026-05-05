import os
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SemanticSearchEngine:
    def __init__(self, collection_name: str = "document_collection"):
        """
        Inisialisasi Semantic Search Engine menggunakan ChromaDB dan Sentence-Transformers.
        """
        # Set up Embedding Model (default to a good multilingual model if not in .env)
        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.embedding_model = SentenceTransformer(model_name)
        
        # Set up ChromaDB Client (Persistent storage)
        db_path = os.getenv("CHROMA_DB_DIR", "data/chroma_db")
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"} # Use cosine similarity for text search
        )

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using SentenceTransformer"""
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        """
        Menambahkan dokumen ke dalam Vector Database.
        """
        if not documents:
            return
            
        embeddings = self._get_embeddings(documents)
        
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Berhasil menambahkan {len(documents)} dokumen ke ChromaDB.")

    def search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Mencari dokumen yang secara semantik mirip dengan query.
        """
        query_embedding = self._get_embeddings([query])
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        return results

# Example usage:
# engine = SemanticSearchEngine()
# engine.add_documents(["Anggaran pendidikan tahun 2024 meningkat"], [{"source": "news"}], ["doc_1"])
# results = engine.search("Berapa dana sekolah tahun depan?", top_k=1)
