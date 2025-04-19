class GeminiClient:
    """Client for interacting with Google's Gemini API."""
    
    def __init__(self, api_key: str):
        """Initialize the Gemini client with API key."""
        try:
            import google.generativeai as genai
            self.genai = genai
            self.api_key = api_key
            genai.configure(api_key=api_key)
            
            # Try to find valid model
            try:
                models = genai.list_models()
                # Filter for Gemini models, prioritizing 1.5 versions
                gemini_models = [model.name for model in models if "gemini" in model.name.lower()]
                gemini_1_5_models = [model for model in gemini_models if "1.5" in model]
                
                if gemini_1_5_models:
                    # Prioritize flash for faster responses
                    if any("flash" in model for model in gemini_1_5_models):
                        self.model_name = next(model for model in gemini_1_5_models if "flash" in model)
                    else:
                        self.model_name = gemini_1_5_models[0]
                elif gemini_models:
                    self.model_name = gemini_models[0]
                else:
                    # Default to the recommended model
                    self.model_name = "gemini-1.5-flash"
                
                st.sidebar.success(f"Using Gemini model: {self.model_name}")
            except Exception as e:
                st.sidebar.warning(f"Error listing models: {e}")
                # Default to the recommended model
                self.model_name = "gemini-1.5-flash"
                
            self.model = genai.GenerativeModel(self.model_name)
        except ImportError:
            st.error("Please install google-generativeai: pip install google-generativeai")
            self.model = None
        except Exception as e:
            st.error(f"Error initializing Gemini: {e}")
            self.model = None
    
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
            error_msg = f"Gemini API error: {e}"
            st.error(error_msg)
            return error_msg
        
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
            error_msg = f"Gemini API error: {e}"
            st.error(error_msg)
            yield error_msg
