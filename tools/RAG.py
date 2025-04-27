import os
from typing import List, Dict, Tuple
import torch
from torch import nn
import PyPDF2
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pathlib import Path
import json
import tempfile
import atexit
import shutil
from logging import Logger as logger

class PDFProcessor:
    """Processes PDF documents for RAG system."""
    
    def __init__(self, pdf_folder: str, cache_dir: Path):
        """
        Initialize PDF processor.
        
        Args:
            pdf_folder (str): Path to folder containing PDF documents
            cache_dir (Path): Path to temporary cache directory
        """
        self.pdf_folder = Path(pdf_folder)
        self.cache_dir = cache_dir
        
    def read_pdf(self, pdf_path: Path) -> str:
        """Extract text from a PDF file."""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 1536) -> List[str]:
        """Split text into chunks of approximately equal size."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            if current_size + len(word) > chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
                current_size += len(word) + 1
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks
    
    def get_document_hash(self, pdf_path: Path) -> str:
        """Generate a hash of the PDF file for caching."""
        import hashlib
        with open(pdf_path, 'rb') as file:
            return hashlib.md5(file.read()).hexdigest()
    
    def process_documents(self) -> Dict[str, List[str]]:
        """Process all PDFs in the folder and return chunks by document."""
        document_chunks = {}
        
        for pdf_file in self.pdf_folder.glob("*.pdf"):
            doc_hash = self.get_document_hash(pdf_file)
            cache_file = self.cache_dir / f"{pdf_file.stem}_{doc_hash}_chunks.json"
            
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    document_chunks[pdf_file.stem] = json.load(f)
            else:
                text = self.read_pdf(pdf_file)
                chunks = self.chunk_text(text)
                document_chunks[pdf_file.stem] = chunks
                
                # Cache the chunks
                with open(cache_file, 'w') as f:
                    json.dump(chunks, f)
            
        return document_chunks

class RAGEmbedding:
    """Handles document embedding for RAG system."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", cache_dir: Path = None):
        """
        Initialize embedding model.
        
        Args:
            model_name (str): Name of the pre-trained model to use
            cache_dir (Path): Path to temporary cache directory
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.cache_dir = cache_dir
        
    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embeddings for a piece of text."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        return embeddings[0]
    
    def embed_chunks(self, doc_id: str, chunks: List[str], doc_hash: str) -> np.ndarray:
        """Generate embeddings for multiple text chunks with caching."""
        cache_file = self.cache_dir / f"{doc_id}_{doc_hash}_embeddings.npy"
        
        if cache_file.exists():
            return np.load(cache_file)
        
        embeddings = []
        for chunk in chunks:
            embedding = self.embed_text(chunk)
            embeddings.append(embedding)
            
        embeddings_array = np.stack(embeddings)
        np.save(cache_file, embeddings_array)
        return embeddings_array

class RAGRetriever:
    """Handles document retrieval for RAG system."""
    
    def __init__(self, top_k: int = 3):
        """
        Initialize retriever.
        
        Args:
            top_k (int): Number of most relevant chunks to retrieve
        """
        self.top_k = top_k
        self.document_chunks = {}
        self.chunk_embeddings = {}
        
    def add_document_embeddings(self, doc_id: str, chunks: List[str], embeddings: np.ndarray):
        """Add document chunks and their embeddings to the retriever."""
        self.document_chunks[doc_id] = chunks
        self.chunk_embeddings[doc_id] = embeddings
        
    def retrieve(self, query_embedding: np.ndarray) -> List[Tuple[str, str, float]]:
        """
        Retrieve most relevant document chunks.
        
        Returns:
            List of tuples containing (doc_id, chunk_text, similarity_score)
        """
        results = []
        
        for doc_id, embeddings in self.chunk_embeddings.items():
            similarities = cosine_similarity([query_embedding], embeddings)[0]
            chunk_scores = list(zip(self.document_chunks[doc_id], similarities))
            results.extend([(doc_id, chunk, score) for chunk, score in chunk_scores])
            
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:self.top_k]

class RAGModule:
    """Main RAG module for integration with ONI system."""
    
    def __init__(self, pdf_folder: str, top_k: int = 3):
        """
        Initialize RAG module.
        
        Args:
            pdf_folder (str): Path to folder containing PDF documents
            top_k (int): Number of chunks to retrieve
        """
        # Create a temporary directory that will be cleaned up on exit
        self.temp_dir = Path(tempfile.mkdtemp(prefix='rag_cache_'))
        atexit.register(self.cleanup)
        
        self.pdf_processor = PDFProcessor(pdf_folder, self.temp_dir)
        self.embedder = RAGEmbedding(cache_dir=self.temp_dir)
        self.retriever = RAGRetriever(top_k=top_k)
        self.initialize_documents()
        
    def cleanup(self):
        """Clean up temporary directory on exit."""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Error cleaning up temporary directory: {e}")
        
    def initialize_documents(self):
        """Process and embed all documents in the PDF folder."""
        document_chunks = self.pdf_processor.process_documents()
        
        for doc_id, chunks in document_chunks.items():
            pdf_path = self.pdf_processor.pdf_folder / f"{doc_id}.pdf"
            doc_hash = self.pdf_processor.get_document_hash(pdf_path)
            embeddings = self.embedder.embed_chunks(doc_id, chunks, doc_hash)
            self.retriever.add_document_embeddings(doc_id, chunks, embeddings)
            
    def query(self, text: str) -> List[Tuple[str, str, float]]:
        """
        Query the RAG system with text.
        
        Returns:
            List of relevant document chunks with their scores
        """
        query_embedding = self.embedder.embed_text(text)
        return self.retriever.retrieve(query_embedding)
class ONIWithRAG(ONI):
    def __init__(self, pdf_folder: str):
        """
        Initialize ONI system with RAG capabilities.
        
        Args:
            pdf_folder (str): Path to folder containing PDF documents
        """
        super().__init__()
        self.rag = RAGModule(pdf_folder)
        
    def run(self, text=None, image=None, show_thinking=False):
        if text is None:
            raise ValueError("`text` (user input) must be provided.")
            
        try:
            # Get relevant document chunks from RAG
            relevant_chunks = self.rag.query(text)
            
            if show_thinking and relevant_chunks:
                print("\n=== Retrieved Documents ===")
                for doc_id, chunk, _ in relevant_chunks:
                    print(f"\nDocument: {doc_id}")
                    print(f"Content: {chunk[:200]}...")
            
            # Create enhanced text with retrieved context
            if relevant_chunks:
                context = "\n".join([f"From {doc_id}: {chunk}" for doc_id, chunk, _ in relevant_chunks])
                enhanced_text = f"Context:\n{context}\n\nOriginal Query: {text}"
            else:
                enhanced_text = text
                
            # Call parent class's run method with enhanced text
            return super().run(enhanced_text, image, show_thinking)
            
        except Exception as e:
            logger.error(f"Error in ONIWithRAG.run: {e}")
            return "I encountered an error processing your request. Please try again."
