class GeminiClient:
    """Client for interacting with Google's Gemini API."""

    def __init__(self, api_key: str):
        """Initialize the Gemini client with API key."""
        try:
            import google.generativeai as genai
            self.genai = genai
            genai.configure(api_key=api_key)
            self.model_name = "gemini-1.5-flash"  # Safe default
            self.model = genai.GenerativeModel(self.model_name)
        except ImportError:
            raise ImportError("Please install google-generativeai: pip install google-generativeai")
        except Exception as e:
            raise RuntimeError(f"Error initializing Gemini: {e}")

    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text response from Gemini."""
        if not self.model:
            return "Gemini not available"

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"temperature": temperature}
            )
            return response.text
        except Exception as e:
            return f"Gemini API error: {e}"

    def generate_stream(self, prompt: str, temperature: float = 0.7):
        """Stream text response from Gemini."""
        if not self.model:
            yield "Gemini not available"
            return

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
            yield f"Gemini API error: {e}"
