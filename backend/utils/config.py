# Configuration Management
# Load environment variables on-demand

import os
from typing import Optional

class Config:
    """Application configuration with lazy loading"""
    
    _instance = None
    _loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._loaded:
            self._load_config()
            self._loaded = True
    
    def _load_config(self):
        """Load configuration from environment"""
        # Only load dotenv when needed
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        
        # Flask config
        self.SECRET_KEY = os.getenv('SECRET_KEY', 'data_viz_secret_key_render_optimized')
        self.UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
        self.MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB
        
        # Together AI config
        self.TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY', '')
        self.TOGETHER_TIMEOUT = int(os.getenv('TOGETHER_TIMEOUT', 30))  # 30s timeout
        
        # Render optimizations
        self.MAX_ROWS_LIMIT = int(os.getenv('MAX_ROWS_LIMIT', 100000))  # Limit data size
        self.CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 10000))  # Process in chunks
        self.ENABLE_MEMORY_MONITORING = os.getenv('ENABLE_MEMORY_MONITORING', 'true').lower() == 'true'
        
        # Debug mode
        self.DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
        
        # Create upload folder
        os.makedirs(self.UPLOAD_FOLDER, exist_ok=True)
    
    @property
    def has_together_api_key(self) -> bool:
        """Check if Together API key is configured"""
        return bool(self.TOGETHER_API_KEY and self.TOGETHER_API_KEY.strip())

# Singleton instance
config = Config()
