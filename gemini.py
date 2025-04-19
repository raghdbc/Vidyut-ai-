import google.generativeai as genai
from typing import Iterator, Dict, Optional

class GeminiClient:
    """Client for interacting with Google's Gemini API."""
    
    def __init__(self, api_key: str):
        """Initialize the Gemini client with API key.
        
        Args:
            api_key: The API key for Gemini access
        """
        self.api_key = api_key
        genai.configure(api_key=api_key)
        
        # Try to list available models to find valid model names
        try:
            models = genai.list_models()
            # Find available Gemini models
            gemini_models = [model.name for model in models if "gemini" in model.name.lower()]
            
            # Use the first available Gemini model or default to gemini-1.5-pro
            if gemini_models:
                self.model_name = gemini_models[0]
            else:
                self.model_name = "gemini-1.5-flash"  # Updated default model name
                
            print(f"Using Gemini model: {self.model_name}")
            self.model = genai.GenerativeModel(self.model_name)
            
        except Exception as e:
            # Fallback to a known model name pattern if listing fails
            print(f"Error listing models: {e}. Using default model.")
            self.model_name = "gemini-1.5-pro"  # Updated default model name
            self.model = genai.GenerativeModel(self.model_name)
    
    def list_available_models(self):
        """List all available models from the Gemini API"""
        try:
            models = genai.list_models()
            return [model.name for model in models]
        except Exception as e:
            print(f"Error listing models: {e}")
            return []
        
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text response from Gemini for the given prompt.
        
        Args:
            prompt: The input text prompt
            temperature: Controls randomness (higher = more random)
            
        Returns:
            The generated text response
        """
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"temperature": temperature}
            )
            return response.text
        except Exception as e:
            error_msg = f"Gemini API error: {e}"
            print(error_msg)
            return error_msg
        
    def generate_stream(self, prompt: str, temperature: float = 0.7) -> Iterator[str]:
        """Stream text response from Gemini for the given prompt.
        
        Args:
            prompt: The input text prompt
            temperature: Controls randomness (higher = more random)
            
        Returns:
            Iterator yielding chunks of generated text
        """
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"temperature": temperature},
                stream=True
            )
            
            for chunk in response:
                if hasattr(chunk, 'text') and chunk.text:
                    yield chunk.text
        except Exception as e:
            error_msg = f"Gemini API error: {e}"
            print(error_msg)
            yield error_msg
