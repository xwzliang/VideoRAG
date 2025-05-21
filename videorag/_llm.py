import numpy as np
import json

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

from ._utils import compute_args_hash, wrap_embedding_func_with_attrs
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
        # Use default Ollama port for embeddings
        global_ollama_client = AsyncClient(host="http://localhost:11434")
    return global_ollama_client

def get_deepseek_async_client_instance():
    global global_deepseek_client
    if global_deepseek_client is None:
        # Use DeepSeek's Ollama-compatible API
        global_deepseek_client = AsyncClient(host="http://localhost:8001")
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
    wait=wait_exponential(multiplier=1, min=4, max=10),
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

async def ollama_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # Initialize the DeepSeek client for text generation
    client = get_deepseek_async_client_instance()

    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None and if_cache_return["return"] is not None:
            return if_cache_return["return"]

    # Format the chat history into a single prompt string
    formatted_prompt = ""
    for msg in messages:
        if msg["role"] == "system":
            formatted_prompt += f"System: {msg['content']}\n"
        elif msg["role"] == "user":
            formatted_prompt += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            formatted_prompt += f"Assistant: {msg['content']}\n"
    
    # Send the request to DeepSeek using streaming
    full_response = ""
    try:
        # First await the generate coroutine to get the stream
        stream = await client.generate(
            model=model,
            prompt=formatted_prompt,
            system=system_prompt,
            stream=True
        )
        
        # Then iterate over the stream
        async for chunk in stream:
            if chunk.get('response'):
                full_response += chunk['response']
    except Exception as e:
        print(f"Error in streaming response: {str(e)}")
        raise
    
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": full_response, "model": model}}
        )
        await hashing_kv.index_done_callback()

    return full_response


async def ollama_complete(model_name, prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await ollama_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages
    )

async def ollama_mini_complete(model_name, prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await ollama_complete_if_cache(
        # "deepseek-r1:latest",  # For now select your model
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages
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
