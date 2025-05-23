import numpy as np
import json
import asyncio
import time
from typing import Optional, List, Dict, Any, Union
import httpx

from openai import AsyncOpenAI, AsyncAzureOpenAI, APIConnectionError, RateLimitError
from ollama import AsyncClient
from dataclasses import asdict, dataclass, field

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import os

from ._utils import compute_args_hash, wrap_embedding_func_with_attrs, logger
from .base import BaseKVStorage
from ._utils import EmbeddingFunc

global_openai_async_client = None
global_azure_openai_async_client = None
global_ollama_client = None
global_deepseek_client = None

def get_openai_async_client_instance():
    global global_openai_async_client
    if global_openai_async_client is None:
        global_openai_async_client = AsyncOpenAI()
    return global_openai_async_client


def get_azure_openai_async_client_instance():
    global global_azure_openai_async_client
    if global_azure_openai_async_client is None:
        global_azure_openai_async_client = AsyncAzureOpenAI()
    return global_azure_openai_async_client

def get_ollama_async_client_instance():
    global global_ollama_client
    if global_ollama_client is None:
        # Use default Ollama port for embeddings with increased timeout
        global_ollama_client = AsyncClient(host="http://localhost:11434", timeout=1200.0)  # 20 minutes timeout
    return global_ollama_client

def get_deepseek_async_client_instance():
    global global_deepseek_client
    if global_deepseek_client is None:
        # Use DeepSeek's Ollama-compatible API with increased timeout
        global_deepseek_client = AsyncClient(host="http://localhost:8001", timeout=1200.0)  # 20 minutes timeout
    return global_deepseek_client

# Setup LLM Configuration.
@dataclass
class LLMConfig:
    # To be set
    embedding_func_raw: callable
    embedding_model_name: str
    embedding_dim: int
    embedding_max_token_size: int
    embedding_batch_num: int    
    embedding_func_max_async: int 
    query_better_than_threshold: float
    
    best_model_func_raw: callable
    best_model_name: str    
    best_model_max_token_size: int
    best_model_max_async: int
    
    cheap_model_func_raw: callable
    cheap_model_name: str
    cheap_model_max_token_size: int
    cheap_model_max_async: int

    # Assigned in post init
    embedding_func: EmbeddingFunc  = None    
    best_model_func: callable = None    
    cheap_model_func: callable = None
    

    def __post_init__(self):
        embedding_wrapper = wrap_embedding_func_with_attrs(
            embedding_dim = self.embedding_dim,
            max_token_size = self.embedding_max_token_size,
            model_name = self.embedding_model_name)
        self.embedding_func = embedding_wrapper(self.embedding_func_raw)
        self.best_model_func = lambda prompt, *args, **kwargs: self.best_model_func_raw(
            self.best_model_name, prompt, *args, **kwargs
        )

        self.cheap_model_func = lambda prompt, *args, **kwargs: self.cheap_model_func_raw(
            self.cheap_model_name, prompt, *args, **kwargs
        )

##### OpenAI Configuration
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=8, max=30),  # Increased wait times
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def openai_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = get_openai_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        # NOTE: I update here to avoid the if_cache_return["return"] is None
        if if_cache_return is not None and if_cache_return["return"] is not None:
            return if_cache_return["return"]

    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": model}}
        )
        await hashing_kv.index_done_callback()
    return response.choices[0].message.content


async def gpt_4o_complete(
        model_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

async def gpt_4o_mini_complete(
        model_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def openai_embedding(model_name: str, texts: list[str]) -> np.ndarray:
    openai_async_client = get_openai_async_client_instance()
    response = await openai_async_client.embeddings.create(
        model=model_name, input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])

openai_config = LLMConfig(
    embedding_func_raw = openai_embedding,
    embedding_model_name = "text-embedding-3-small",
    embedding_dim = 1536,
    embedding_max_token_size  = 8192,
    embedding_batch_num = 32,
    embedding_func_max_async = 16,
    query_better_than_threshold = 0.2,

    # LLM        
    best_model_func_raw = gpt_4o_complete,
    best_model_name = "gpt-4o",    
    best_model_max_token_size = 32768,
    best_model_max_async = 16,
        
    cheap_model_func_raw = gpt_4o_mini_complete,
    cheap_model_name = "gpt-4o-mini",
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 16
)

openai_4o_mini_config = LLMConfig(
    embedding_func_raw = openai_embedding,
    embedding_model_name = "text-embedding-3-small",
    embedding_dim = 1536,
    embedding_max_token_size  = 8192,
    embedding_batch_num = 32,
    embedding_func_max_async = 16,
    query_better_than_threshold = 0.2,

    # LLM        
    best_model_func_raw = gpt_4o_mini_complete,
    best_model_name = "gpt-4o-mini",    
    best_model_max_token_size = 32768,
    best_model_max_async = 16,
        
    cheap_model_func_raw = gpt_4o_mini_complete,
    cheap_model_name = "gpt-4o-mini",
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 16
)

###### Azure OpenAI Configuration
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def azure_openai_complete_if_cache(
    deployment_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    azure_openai_client = get_azure_openai_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(deployment_name, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        # NOTE: I update here to avoid the if_cache_return["return"] is None
        if if_cache_return is not None and if_cache_return["return"] is not None:
            return if_cache_return["return"]

    response = await azure_openai_client.chat.completions.create(
        model=deployment_name, messages=messages, **kwargs
    )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {
                args_hash: {
                    "return": response.choices[0].message.content,
                    "model": deployment_name,
                }
            }
        )
        await hashing_kv.index_done_callback()
    return response.choices[0].message.content


async def azure_gpt_4o_complete(
        model_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await azure_openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def azure_gpt_4o_mini_complete(
        model_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await azure_openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )



@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def azure_openai_embedding(model_name: str, texts: list[str]) -> np.ndarray:
    azure_openai_client = get_azure_openai_async_client_instance()
    response = await azure_openai_client.embeddings.create(
        model=model_name, input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])


azure_openai_config = LLMConfig(
    embedding_func_raw = azure_openai_embedding,
    embedding_model_name = "text-embedding-3-small",
    embedding_dim = 1536,
    embedding_max_token_size = 8192,    
    embedding_batch_num = 32,
    embedding_func_max_async = 16,
    query_better_than_threshold = 0.2,

    best_model_func_raw = azure_gpt_4o_complete,
    best_model_name = "gpt-4o",    
    best_model_max_token_size = 32768,
    best_model_max_async = 16,

    cheap_model_func_raw  = azure_gpt_4o_mini_complete,
    cheap_model_name = "gpt-4o-mini",
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 16
)


######  Ollama configuration

def clean_response_text(text: str) -> str:
    """Clean up response text by removing unwanted markers and thinking process."""
    if "</think>\n" in text:
        return text.split("</think>\n")[-1].strip()
    return text.strip()

async def ollama_complete_if_cache(
    prompt: str,
    model_name: str = "deepseek-coder",
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[Dict[str, str]]] = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    hashing_kv=None,
    use_cache: bool = True,
    max_retries: int = 5,  # Increased from 3
    retry_delay: float = 2.0,  # Increased from 1.0
) -> str:
    """Complete a prompt using DeepSeek service with caching and retry logic."""
    # Check cache first
    if hashing_kv is not None and use_cache:
        cache_key = f"{model_name}:{prompt}:{system_prompt}:{json.dumps(history_messages) if history_messages else ''}"
        cached_response = await hashing_kv.get_by_id(cache_key)
        if cached_response is not None:
            return clean_response_text(cached_response)

    # Format the prompt
    formatted_prompt = prompt
    if system_prompt:
        formatted_prompt = f"System: {system_prompt}\n\nUser: {prompt}"
    else:
        formatted_prompt = f"User: {prompt}"

    # Prepare request data
    data = {
        "model": "DeepSeek-R1",  # Use the correct model name
        "prompt": formatted_prompt,
        "stream": True,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    # Log request details
    logger.info(f"Model name: {data['model']}")
    logger.info(f"Original prompt: {prompt}")
    logger.info(f"Formatted prompt: {formatted_prompt}")
    logger.info(f"Full request data: {json.dumps(data, indent=2)}")

    for attempt in range(max_retries):
        try:
            # Create client with increased timeout to match server's 20-minute timeout
            async with httpx.AsyncClient(timeout=1200.0) as client:  # 20 minutes
                response = await client.post(
                    "http://localhost:8001/api/generate",
                    json=data,
                    headers={
                        "accept": "application/json",
                        "Content-Type": "application/json"
                    }
                )
                response.raise_for_status()
                
                # Handle streaming response
                full_response = ""
                async for line in response.aiter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            if "response" in chunk:
                                # Only append if it's not a JSON object
                                if not (chunk["response"].startswith("{") and chunk["response"].endswith("}")):
                                    full_response += chunk["response"]
                        except json.JSONDecodeError:
                            continue
                
                response_text = clean_response_text(full_response.strip())
                
                # Validate response
                # if not response_text or len(response_text) < 2:
                #     raise ValueError("Empty or too short response received")
                
                # Check if response contains JSON-like content
                if (response_text.startswith("{") and response_text.endswith("}")) or \
                   (response_text.startswith("[") and response_text.endswith("]")):
                    raise ValueError("Response contains unexpected JSON content")
                
                logger.info(f"Received response from DeepSeek:\n{response_text}")
                
                # Cache the response
                if hashing_kv is not None and use_cache:
                    await hashing_kv.upsert({cache_key: response_text})
                
                return response_text
                    
        except (httpx.ReadError, httpx.ConnectError, httpx.TimeoutException) as e:
            if attempt < max_retries - 1:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
        except Exception as e:
            error_msg = f"Unexpected error during DeepSeek request: {str(e)}"
            logger.error(error_msg)
            if attempt < max_retries - 1:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                raise

async def ollama_mini_complete(
    model_name: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[Dict[str, str]]] = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    hashing_kv=None,
    use_cache: bool = True,
) -> str:
    """Wrapper for ollama_complete_if_cache with mini model settings."""
    logger.info(f"ollama_mini_complete called with prompt: {prompt}")
    return await ollama_complete_if_cache(
        prompt=prompt,
        model_name=model_name,
        system_prompt=system_prompt,
        history_messages=history_messages,
        max_tokens=max_tokens,
        temperature=temperature,
        hashing_kv=hashing_kv,
        use_cache=use_cache
    )

async def ollama_complete(model_name: str, prompt: str, system_prompt=None, history_messages=[], **kwargs) -> str:
    logger.info(f"ollama_complete called with prompt: {prompt}")
    return await ollama_complete_if_cache(
        prompt=prompt,
        model_name=model_name,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs
    )

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def ollama_embedding(model_name: str, texts: list[str]) -> np.ndarray:
    # Initialize the Ollama client for embeddings
    ollama_client = get_ollama_async_client_instance()

    # Send the request to Ollama for embeddings
    response = await ollama_client.embed(
        model=model_name,  
        input=texts
    )

    # Extract embeddings from the response
    embeddings = response['embeddings']

    return np.array(embeddings)

ollama_config = LLMConfig(
    embedding_func_raw = ollama_embedding,  # Use Ollama's native API for embeddings
    embedding_model_name = "nomic-embed-text",
    embedding_dim = 768,
    embedding_max_token_size=8192,
    embedding_batch_num = 1,
    embedding_func_max_async = 1,
    query_better_than_threshold = 0.2,
    best_model_func_raw = ollama_complete,
    best_model_name = "deepseek-coder",  # Use DeepSeek through Ollama-compatible API
    best_model_max_token_size = 32768,
    best_model_max_async = 1,
    cheap_model_func_raw = ollama_mini_complete,
    cheap_model_name = "deepseek-coder",  # Use DeepSeek through Ollama-compatible API
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 1
)
