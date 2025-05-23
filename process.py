import os
import sys
import logging
import warnings
import multiprocessing
import subprocess

# Get the CUDA library path
import nvidia.cudnn.lib
cudnn_lib_path = os.path.dirname(nvidia.cudnn.lib.__file__)

# If LD_LIBRARY_PATH is not set correctly, restart the script with the correct path
if not os.environ.get('LD_LIBRARY_PATH') or cudnn_lib_path not in os.environ['LD_LIBRARY_PATH']:
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = cudnn_lib_path
    print(f"Restarting with LD_LIBRARY_PATH={cudnn_lib_path}")
    sys.exit(subprocess.call([sys.executable] + sys.argv, env=env))

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)

from videorag._llm import ollama_config
from videorag import VideoRAG, QueryParam

# Configure local models
local_model_config = ollama_config
local_model_config.embedding_model_name = "nomic-embed-text"  # Local embedding model
local_model_config.best_model_name = "deepseek-coder"  # DeepSeek model name (without :latest)
local_model_config.cheap_model_name = "deepseek-coder"  # Using same model for both

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    # Please enter your video file path in this list; there is no limit on the length.
    # Here is an example; you can use your own videos instead.
    video_paths = [
        '~/videos/Batman_Begins_2005_BluRay_1080p_x265_10bit_2Audio_MNHD-FRDS.mkv',
    ]
    video_paths = [
        os.path.expanduser(p) for p in video_paths
    ]
    videorag = VideoRAG(llm=local_model_config, working_dir=os.path.expanduser("~/videos/videorag-workdir"))
    
    # Load Qwen-VL model for vision tasks
    videorag.caption_model = "Qwen/Qwen-VL-Chat"  # This will be loaded when needed
    
    # Insert videos
    videorag.insert_video(video_path_list=video_paths)
    
    # To regenerate a specific video:
    # videorag.regenerate_video(video_paths[0])