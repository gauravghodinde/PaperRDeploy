"""
LLM Factory.

Centralized LLM initialization for switching between providers.
Supports: Groq (default), Gemini, DeepSeek, OpenAI, Mistral
"""

import os
from langchain_core.language_models.chat_models import BaseChatModel
from src.utils.config import settings


def get_llm(temperature: float = 0.2, provider: str = None) -> BaseChatModel:
    """
    Get an LLM instance based on provider.
    
    Args:
        temperature: Model temperature (0.0-1.0)
        provider: Override provider ("groq", "gemini", or "deepseek"). 
                  If None, uses settings.LLM_PROVIDER.
    
    Returns:
        LangChain chat model instance
    """
    provider = provider or settings.LLM_PROVIDER
    
    if provider.lower() == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        api_key = settings.GOOGLE_API_KEY
        if not api_key or api_key == "your-google-key-here":
            raise ValueError(
                "❌ GOOGLE_API_KEY not configured!\n"
                "Get your key from: https://aistudio.google.com/app/apikey\n"
                "Add it to your .env file: GOOGLE_API_KEY=AIza..."
            )
        
        print(f"✓ Using Gemini API (key: {api_key[:10]}...{api_key[-4:]})")
        try:
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite",
                temperature=temperature,
                google_api_key=api_key
            )
        except Exception as e:
            print(f"⚠️ Failed to initialize Gemini: {e}")
            print("🔄 Falling back to Groq...")
            from langchain_groq import ChatGroq
            os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY
            return ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=temperature
            )
    
    elif provider.lower() == "deepseek":
        from langchain_openai import ChatOpenAI
        
        api_key = settings.DEEPSEEK_API_KEY
        if not api_key:
            raise ValueError(
                "❌ DEEPSEEK_API_KEY not configured!\n"
                "Get your key from: https://platform.deepseek.com/\n"
                "Add it to your .env file: DEEPSEEK_API_KEY=sk-..."
            )
        
        print(f"✓ Using DeepSeek API (key: {api_key[:10]}...{api_key[-4:]})")
        return ChatOpenAI(
            model="deepseek-chat",
            api_key=api_key,
            base_url="https://api.deepseek.com",
            temperature=temperature
        )
        
    elif provider.lower() == "openai":
        from langchain_openai import ChatOpenAI
        
        api_key = settings.OPENAI_API_KEY
        if not api_key:
            raise ValueError(
                "❌ OPENAI_API_KEY not configured!\n"
                "Add it to your .env file: OPENAI_API_KEY=sk-..."
            )
            
        print(f"✓ Using OpenAI API (key: {api_key[:10]}...{api_key[-4:]})")
        return ChatOpenAI(
            model="gpt-4o",  # Default for OpenAI, caller can override
            api_key=api_key,
            temperature=temperature
        )

    elif provider.lower() == "mistral":
        from langchain_mistralai import ChatMistralAI
        
        # We need to add MISTRAL_API_KEY to settings, fallback to os.environ
        api_key = getattr(settings, "MISTRAL_API_KEY", os.environ.get("MISTRAL_API_KEY", ""))
        if not api_key:
            raise ValueError(
                "❌ MISTRAL_API_KEY not configured!\n"
                "Add it to your .env file: MISTRAL_API_KEY=..."
            )
            
        print(f"✓ Using Mistral API (key: {api_key[:10]}...{api_key[-4:]})")
        return ChatMistralAI(
            model="mistral-large-latest",
            mistral_api_key=api_key,
            temperature=temperature
        )
    
    else:
        # Default: Groq (FREE tier)
        from langchain_groq import ChatGroq
        
        os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            # model="openai/gpt-oss-120b",
            temperature=temperature
        )

