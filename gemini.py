class GeminiClient:
    """Client for interacting with Google's Gemini API."""

    def __init__(self, api_key: str):
        """Initialize the Gemini client with API key."""
        try:
            import google.generativeai as genai
            self.genai = genai
            self.api_key = api_key
            genai.configure(api_key=api_key)

            # Optional Streamlit support
            try:
                import streamlit as st
                self.st = st
            except ImportError:
                self.st = None

            # Select and set model
            self.model_name = self._select_model()
            self.model = genai.GenerativeModel(self.model_name)

            if self.st:
                self.st.sidebar.success(f"Using Gemini model: {self.model_name}")
            else:
                print(f"[INFO] Using Gemini model: {self.model_name}")

        except ImportError:
            if self.st:
                self.st.error("Please install google-generativeai: pip install google-generativeai")
            else:
                print("[ERROR] Missing google-generativeai. Run: pip install google-generativeai")
            self.model = None
        except Exception as e:
            if self.st:
                self.st.error(f"Error initializing Gemini: {e}")
            else:
                print(f"[ERROR] Error initializing Gemini: {e}")
            self.model = None

    def _select_model(self) -> str:
        """Select the best available Gemini model, prioritizing 1.5 flash."""
        try:
            models = self.genai.list_models()
            gemini_models = [model.name for model in models if "gemini" in model.name.lower()]
            gemini_models = [m for m in gemini_models if not m.startswith("models/gemini-pro-vision")]  # Avoid deprecated

            gemini_1_5_models = [m for m in gemini_models if "1.5" in m]
            if gemini_1_5_models:
                return next((m for m in gemini_1_5_models if "flash" in m), gemini_1_5_models[0])
            elif gemini_models:
                return gemini_models[0]
        except Exception as e:
            if self.st:
                self.st.sidebar.warning(f"Error listing models: {e}")
            else:
                print(f"[WARN] Error listing models: {e}")
        return "gemini-1.5-flash"

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
            msg = f"Gemini API error: {e}"
            if self.st:
                self.st.error(msg)
            else:
                print(f"[ERROR] {msg}")
            return msg

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
            msg = f"Gemini API error: {e}"
            if self.st:
                self.st.error(msg)
            else:
                print(f"[ERROR] {msg}")
            yield msg
