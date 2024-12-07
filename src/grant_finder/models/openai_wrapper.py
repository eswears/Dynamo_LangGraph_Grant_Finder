from langchain_openai import ChatOpenAI
from typing import List, Dict
from .base import BaseLanguageModel

class OpenAIWrapper(BaseLanguageModel):
    def __init__(self, **kwargs):
        self.model = ChatOpenAI(**kwargs)
        
    def generate(self, prompt: str, **kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)
        
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        response = self.model.invoke(messages)
        return response.content 