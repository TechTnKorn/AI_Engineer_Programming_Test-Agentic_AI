from langchain_openai import AzureChatOpenAI
from langchain_ollama import ChatOllama

import os
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

LLMModel = AzureChatOpenAI | ChatOllama

def select_llm_provider(llm_provider: str=None, temperature: float=None, reasoning_effort: str=None, streaming: bool=True) -> LLMModel: # type: ignore
    """Select LLM by LLM_PROVIDER environment variable"""
    
    if llm_provider is None:
        llm_provider = os.getenv("LLM_PROVIDER")

    llm = None
    
    if llm_provider == "azure":
        llm = AzureChatOpenAI(
            model=os.getenv("AZURE_DEPLOYMENT"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version=os.getenv("AZURE_OPENAI_VERSION"),
            temperature=temperature,
            azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
            streaming=streaming,
            # reasoning_effort=reasoning_effort
            top_p=0.9
        )
    elif llm_provider == "ollama":
        disable_streaming = not streaming
        llm = ChatOllama(
            base_url=os.getenv("OLLAMA_BASE_URL"),
            model=os.getenv("OLLAMA_MODEL"),
            reasoning=reasoning_effort,
            disable_streaming=disable_streaming,
            temperature=temperature,
            top_p=0.9
        )
        
    return llm
