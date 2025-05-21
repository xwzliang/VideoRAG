import os
import logging
import warnings
import multiprocessing

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)

# Please enter your openai key
os.environ["OPENAI_API_KEY"] = ""

from videorag._llm import ollama_config
from videorag import VideoRAG, QueryParam

# Configure local models
local_model_config = ollama_config
local_model_config.embedding_model_name = "nomic-embed-text"  # Local embedding model
local_model_config.best_model_name = "deepseek-coder"  # DeepSeek model name (without :latest)
local_model_config.cheap_model_name = "deepseek-coder"  # Using same model for both

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    query = 'What is the relationship between the girl and the headmistress?'
    param = QueryParam(mode="videorag")
    # if param.wo_reference = False, VideoRAG will add reference to video clips in the response
    param.wo_reference = True

    videorag = VideoRAG(llm=local_model_config, working_dir=f"./videorag-workdir")
    
    # Set Qwen-VL model to be loaded via REST API
    videorag.caption_model = "Qwen/Qwen-VL-Chat"  # This will be loaded via REST API when needed
    
    # response = videorag.query(query=query, param=param)

    response = videorag.regenerate_query(query=query, param=param)

    print(response)