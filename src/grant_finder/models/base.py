from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseLanguageModel(ABC):
    """Base class for language models"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat completion interface"""
        pass 