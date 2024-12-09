from pathlib import Path
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain.storage import LocalFileStore
import json
from typing import Optional, Dict, Any

class PersistenceManager:
    """Manages persistence of chat history and other data"""
    
    def __init__(self, persistence_dir: Path):
        """Initialize persistence manager with directory path"""
        self.persistence_dir = persistence_dir
        self.persistence_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage
        self.store = LocalFileStore(str(persistence_dir))
        
        # Create chat history directory
        self.chat_dir = persistence_dir / "chat_history"
        self.chat_dir.mkdir(exist_ok=True)
        
        # Create sessions directory
        self.sessions_dir = persistence_dir / "sessions"
        self.sessions_dir.mkdir(exist_ok=True)
    
    def get_chat_history(self, session_id: str) -> FileChatMessageHistory:
        """Get chat history for a session"""
        history_file = self.chat_dir / f"{session_id}.json"
        return FileChatMessageHistory(str(history_file))
    
    def save_data(self, key: str, data: bytes) -> None:
        """Save binary data with given key"""
        self.store.mset({key: data})
    
    def load_data(self, key: str) -> bytes:
        """Load binary data for given key"""
        return self.store.mget([key])[0]
    
    def clear_session(self, session_id: str) -> None:
        """Clear chat history and data for a session"""
        history_file = self.chat_dir / f"{session_id}.json"
        if history_file.exists():
            history_file.unlink()
            
    def load_last_session(self) -> Optional[dict]:
        """Load the last session state if it exists"""
        try:
            session_files = list(self.sessions_dir.glob("*.json"))
            if not session_files:
                return None
                
            # Get most recent session file
            latest_session = max(session_files, key=lambda p: p.stat().st_mtime)
            
            with open(latest_session, 'r') as f:
                return json.load(f)
                
        except Exception:
            return None 
    
    def save_context(self, session_id: str, context: Dict[str, Any]) -> None:
        """Save hierarchical context for a session"""
        context_file = self.sessions_dir / f"{session_id}_context.json"
        with open(context_file, 'w') as f:
            json.dump(context, f, indent=2)
    
    def load_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load hierarchical context for a session"""
        context_file = self.sessions_dir / f"{session_id}_context.json"
        if context_file.exists():
            with open(context_file, 'r') as f:
                return json.load(f)
        return None