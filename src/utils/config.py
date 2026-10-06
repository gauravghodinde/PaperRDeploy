"""
Configuration management using Pydantic Settings.
Loads settings from .env file.
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    OPENAI_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    GROQ_API_KEY: str = ""  # FREE tier!
    SEMANTIC_SCHOLAR_API_KEY: str = ""
    DEEPSEEK_API_KEY: str = ""  # DeepSeek API key
    
    # LLM Provider: "groq" (default), "gemini", or "deepseek"
    LLM_PROVIDER: str = "groq"
    
    # Pipeline Settings
    MAX_PAPERS: int = 15
    SUMMARY_WORDS: int = 50
    MAX_PDF_PAGES: int = 2
    
    # Rate Limiting
    API_DELAY_SECONDS: float = 1.0
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
