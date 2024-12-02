# New file: grant_finder/persistence.py - New code in green:
from pathlib import Path
import json
from datetime import datetime
from typing import Optional, Dict, Any
import sqlite3
from contextlib import contextmanager
from .types import GrantFinderState

class PersistenceManager:
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.db_path = storage_dir / "grant_finder.db"
        self.initialize_storage()
    
    def initialize_storage(self):
        """Initialize SQLite database for persistent storage"""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        with self.get_db() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    state JSON,
                    is_active INTEGER
                );
                
                CREATE TABLE IF NOT EXISTS search_history (
                    id INTEGER PRIMARY KEY,
                    session_id TEXT,
                    timestamp TEXT,
                    action TEXT,
                    data JSON,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );
                
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY,
                    session_id TEXT,
                    timestamp TEXT,
                    role TEXT,
                    content TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );
            """)
    
    @contextmanager
    def get_db(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def save_state(self, state: GrantFinderState) -> None:
        """Save current state to persistent storage"""
        with self.get_db() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sessions (session_id, timestamp, state, is_active) VALUES (?, ?, ?, ?)",
                (state.session_id, state.timestamp, json.dumps(state.dict()), 1)
            )
    
    def load_last_session(self) -> Optional[Dict[str, Any]]:
        """Load the most recent active session"""
        with self.get_db() as conn:
            row = conn.execute(
                "SELECT state FROM sessions WHERE is_active = 1 ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            
            if row:
                return json.loads(row['state'])
        return None
    
    def add_chat_message(self, session_id: str, role: str, content: str) -> None:
        """Add a chat message to history"""
        with self.get_db() as conn:
            conn.execute(
                "INSERT INTO chat_history (session_id, timestamp, role, content) VALUES (?, ?, ?, ?)",
                (session_id, datetime.now().isoformat(), role, content)
            )