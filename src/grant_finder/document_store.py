from pathlib import Path
from typing import Dict, List, Optional, Set
import json
import hashlib
from datetime import datetime

from langchain_community.document_loaders import (
    PyPDFLoader, UnstructuredFileLoader, DirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings  # Update this import at the top of the file
from langchain.docstore.document import Document
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain_core.documents import Document

class DocumentStoreManager:
    """Manages document indexing and retrieval for company knowledge base"""
    
    def __init__(
        self,
        docs_dir: Path,
        storage_dir: Path,
        embeddings_model: Optional[OpenAIEmbeddings] = None
    ):
        self.docs_dir = Path(docs_dir)
        if not self.docs_dir.exists():
            raise ValueError(f"Documents directory does not exist: {self.docs_dir}")
        if not self.docs_dir.is_dir():
            raise ValueError(f"Path is not a directory: {self.docs_dir}")
        
        # Create storage directory if it doesn't exist
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Create index directory if it doesn't exist
        self.index_dir = storage_dir / "indices"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_path = storage_dir / "document_metadata.json"
        self.embeddings = embeddings_model or OpenAIEmbeddings()
        
        # Initialize or load document metadata
        self.document_metadata = self._load_metadata()
        
        # Setup document splitting
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Initialize retriever
        self.store = LocalFileStore(self.storage_dir / "chunks")
        self.retriever = self._initialize_retriever()

    def _load_metadata(self) -> Dict:
        """Load or initialize document metadata"""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        return {
            "indexed_files": {},
            "last_update": None
        }

    def _save_metadata(self):
        """Save current document metadata"""
        self.document_metadata["last_update"] = datetime.now().isoformat()
        with open(self.metadata_path, 'w') as f:
            json.dump(self.document_metadata, f, indent=2)

    def _get_file_hash(self, file_path: Path) -> str:
        """Generate hash of file contents"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _initialize_retriever(self) -> ParentDocumentRetriever:
        """Initialize or load the retriever with existing indices"""
        vectorstore = FAISS.load_local(
            self.index_dir / "faiss_index",
            self.embeddings,
            allow_dangerous_deserialization=True
        ) if (self.index_dir / "faiss_index").exists() else FAISS.from_documents(
            [Document(page_content="", metadata={})],  # Initialize empty
            self.embeddings
        )
        
        return ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=self.store,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter,
        )

    def update_document_index(self) -> List[str]:
        """Update indices for new or modified documents"""
        new_or_modified = []
        
        # Scan all documents in directory
        for file_path in self.docs_dir.rglob("*"):
            if not file_path.is_file() or file_path.suffix.lower() not in ['.pdf', '.txt', '.docx']:
                continue
                
            file_hash = self._get_file_hash(file_path)
            rel_path = str(file_path.relative_to(self.docs_dir))
            
            # Check if file is new or modified
            if (rel_path not in self.document_metadata["indexed_files"] or 
                self.document_metadata["indexed_files"][rel_path]["hash"] != file_hash):
                
                # Load and process document
                if file_path.suffix.lower() == '.pdf':
                    loader = PyPDFLoader(str(file_path))
                else:
                    loader = UnstructuredFileLoader(str(file_path))
                
                documents = loader.load()
                
                # Convert documents to bytes for storage
                serializable_docs = []
                for i, doc in enumerate(documents):
                    doc_dict = {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    # Convert to bytes and add as tuple with safe key
                    doc_bytes = json.dumps(doc_dict).encode('utf-8')
                    # Create safe key by replacing spaces and special chars
                    safe_key = rel_path.replace(' ', '_').replace('.', '_').replace('-', '_')
                    safe_key = f"doc_{safe_key}_{i}"
                    serializable_docs.append((safe_key, doc_bytes))
                
                # Add documents to retriever's docstore
                self.retriever.docstore.mset(serializable_docs)
                
                # Add to vectorstore
                self.retriever.vectorstore.add_documents(documents)
                
                # Update metadata
                self.document_metadata["indexed_files"][rel_path] = {
                    "hash": file_hash,
                    "last_indexed": datetime.now().isoformat(),
                    "size": file_path.stat().st_size
                }
                
                new_or_modified.append(rel_path)
        
        # Save updated indices and metadata
        if new_or_modified:
            self.retriever.vectorstore.save_local(self.index_dir / "faiss_index")
            self._save_metadata()
        
        return new_or_modified

    def get_relevant_documents(self, query: str, max_documents: int = 5) -> List[Document]:
        """Retrieve relevant documents for a query"""
        return self.retriever.invoke(query, config={"max_documents": max_documents})

    def get_document_stats(self) -> Dict:
        """Get statistics about indexed documents"""
        return {
            "total_documents": len(self.document_metadata["indexed_files"]),
            "total_size": sum(
                info["size"] for info in self.document_metadata["indexed_files"].values()
            ),
            "last_update": self.document_metadata["last_update"]
        }