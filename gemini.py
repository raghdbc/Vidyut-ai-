import google.generativeai as genai
from typing import Iterator, Dict, Optional

class GeminiClient:
    """Client for interacting with Google's Gemini API."""
    
    def __init__(self, api_key: str):

        self.api_key = api_key
        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel('gemini-pro')
        
    def generate(self, prompt: str, temperature: float = 0.7) -> str:

        response = self.model.generate_content(
            prompt,
            generation_config={"temperature": temperature}
        )
        return response.text
        
    def generate_stream(self, prompt: str, temperature: float = 0.7) -> Iterator[str]:

        response = self.model.generate_content(
            prompt,
            generation_config={"temperature": temperature},
            stream=True
        )
        
        for chunk in response:
            if hasattr(chunk, 'text') and chunk.text:
                yield chunk.text