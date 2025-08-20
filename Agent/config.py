import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the Gemini AI Agent"""
    
    # API Keys - Set default values
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', "AIzaSyCe9Yk344zlvW78DWV60cJscC9--tbNwP0")
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', "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")
    HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY', "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")
    
    # Gemini Model Configuration
    GEMINI_MODEL = "gemini-2.0-flash-exp"  # Using the latest Gemini 2.0 Flash model
    GEMINI_PRO_MODEL = "gemini-1.5-pro"
    
    # Agent Settings
    MAX_TOKENS = 8192
    TEMPERATURE = 0.7
    TOP_P = 0.9
    
    # Data Science Settings
    DEFAULT_SAMPLE_SIZE = 1000
    MAX_DATASET_SIZE = 10000
    
    # File Paths
    UPLOADS_DIR = "uploads"
    OUTPUTS_DIR = "outputs"
    CACHE_DIR = "cache"
    
    @classmethod
    def validate_config(cls):
        """Validate that required configuration is present"""
        if not cls.GEMINI_API_KEY or cls.GEMINI_API_KEY == "your_gemini_api_key_here":
            raise ValueError("GEMINI_API_KEY is required. Please set it in your .env file or update config.py.")
        
        # Create necessary directories
        for directory in [cls.UPLOADS_DIR, cls.OUTPUTS_DIR, cls.CACHE_DIR]:
            os.makedirs(directory, exist_ok=True)
        
        return True
