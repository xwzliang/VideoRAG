import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import requests
from moviepy.video.io.VideoFileClip import VideoFileClip
import gc
from videorag.prompt import PROMPTS
import time

def load_vision_model():
    """Request server to load the vision model."""
    try:
        # First unload LLM model to free up resources
        unload_llm_model()
        
        response = requests.post("http://localhost:8000/load_model")
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error loading vision model: {str(e)}")
        return False

def unload_vision_model():
    """Request server to unload the vision model."""
    try:
        response = requests.post("http://localhost:8000/unload_model")
        response.raise_for_status()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error unloading vision model: {str(e)}")
        return False

def load_llm_model():
    """Request server to load the DeepSeek model for LLM tasks."""
    try:
        # First unload vision model to free up resources
        unload_vision_model()
        
        response = requests.post("http://localhost:8002/load_model")
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error loading DeepSeek model: {str(e)}")
        return False

def unload_llm_model():
    """Request server to unload the DeepSeek model."""
    try:
        response = requests.post("http://localhost:8002/unload_model")
        response.raise_for_status()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error unloading DeepSeek model: {str(e)}")
        return False

def encode_video(video, frame_times):
    frames = []
    for t in frame_times:
        frames.append(video.get_frame(t))
    frames = np.stack(frames, axis=0)
    frames = [Image.fromarray(v.astype('uint8')).resize((1280, 720)) for v in frames]
    return frames
    
async def segment_caption(video_name, video_path, segment_index2name, transcripts, segment_times_info, caption_result, error_queue, working_dir):
    try:
        # Qwen-VL API endpoint
        QWENVL_API_URL = "http://localhost:8000/generate_caption"
        
        # Load model before processing
        if not load_vision_model():
            error_queue.put("Failed to load vision model")
            raise RuntimeError("Failed to load vision model")
            
        try:
            # Load existing captions from storage
            from .._storage import JsonKVStorage
            from .._utils import logger
            
            # Initialize storage for captions with working directory
            caption_storage = JsonKVStorage(namespace="video_captions", global_config={"working_dir": working_dir})
            
            # Check which segments already have captions
            cached_captions = caption_storage._data.get(video_name, {})
            logger.info(f"Found {len(cached_captions)} cached captions for {video_name}")
            
            # Sort segments by index
            sorted_segments = sorted(segment_index2name.keys(), key=int)
            logger.info(f"Processing {len(sorted_segments)} segments in order")
            
            for index in tqdm(sorted_segments, desc=f"Captioning Video {video_name}"):
                # Skip if we already have a caption for this segment
                if str(index) in cached_captions:
                    logger.info(f"Using cached caption for segment {index}")
                    caption_result[index] = cached_captions[str(index)]
                    continue
                
                segment_transcript = transcripts[index]
                # Get segment video path from _cache directory
                # Remove any existing extension from segment_name before adding .mp4
                segment_name = os.path.splitext(segment_index2name[index])[0]
                segment_path = os.path.join(working_dir, "_cache", video_name, f"{segment_name}.mp4")
                
                # Prepare request data for Qwen-VL
                request_data = {
                    "video_path": segment_path,
                    "transcript": segment_transcript,
                    "query": PROMPTS["video_caption"],
                    "fps": 3.0  # Adjust based on your needs
                }
                
                # Make request to Qwen-VL API
                try:
                    response = requests.post(QWENVL_API_URL, json=request_data)
                    response.raise_for_status()
                    segment_caption = response.json()["caption"]
                    
                    # Cache the caption immediately
                    if video_name not in caption_storage._data:
                        caption_storage._data[video_name] = {}
                    caption_storage._data[video_name][str(index)] = segment_caption
                    logger.info(f"Saving caption for segment {index} to storage")
                    await caption_storage.index_done_callback()
                    logger.info(f"Successfully saved caption for segment {index}")
                    
                    caption_result[index] = segment_caption.replace("\n", " ").strip()
                    logger.info(f"Generated and cached caption for segment {index}")
                except requests.exceptions.RequestException as e:
                    error_queue.put(f"Error calling Qwen-VL API:\n {str(e)}")
                    raise RuntimeError
                
        except Exception as e:
            error_queue.put(f"Error in segment_caption:\n {str(e)}")
            raise RuntimeError
                
    except Exception as e:
        error_queue.put(f"Error in segment_caption:\n {str(e)}")
        raise RuntimeError

def merge_segment_information(segment_index2name, segment_times_info, transcripts, captions):
    inserting_segments = {}
    for index in segment_index2name:
        inserting_segments[index] = {"content": None, "time": None}
        # Remove any existing extension from segment_name
        segment_name = os.path.splitext(segment_index2name[index])[0]
        inserting_segments[index]["time"] = '-'.join(segment_name.split('-')[-2:])
        inserting_segments[index]["content"] = f"Caption:\n{captions[index]}\nTranscript:\n{transcripts[index]}\n\n"
        inserting_segments[index]["transcript"] = transcripts[index]
        inserting_segments[index]["frame_times"] = segment_times_info[index]["frame_times"].tolist()
    return inserting_segments
        
def retrieved_segment_caption(caption_model, caption_tokenizer, refine_knowledge, retrieved_segments, video_path_db, video_segments, num_sampled_frames):
    # Qwen-VL API endpoint
    QWENVL_API_URL = "http://localhost:8000/generate_caption"
    
    # Load model before processing
    if not load_vision_model():
        print("Failed to load vision model")
        return {}
        
    try:
        caption_result = {}
        for this_segment in tqdm(retrieved_segments, desc='Captioning Segments for Given Query'):
            video_name = '_'.join(this_segment.split('_')[:-1])
            index = this_segment.split('_')[-1]
            video_path = video_path_db._data[video_name]
            timestamp = video_segments._data[video_name][index]["time"].split('-')
            start_time, end_time = eval(timestamp[0]), eval(timestamp[1])
            
            segment_transcript = video_segments._data[video_name][index]["transcript"]
            
            # Prepare request data for Qwen-VL with refined query
            request_data = {
                "video_path": video_path,
                "start_time": float(start_time),
                "end_time": float(end_time),
                "transcript": segment_transcript,
                "query": PROMPTS["query_video_caption"].format(refine_knowledge=refine_knowledge),
                "fps": 5.0  # Adjust based on your needs
            }
            
            # Make request to Qwen-VL API
            try:
                response = requests.post(QWENVL_API_URL, json=request_data)
                response.raise_for_status()
                this_caption = response.json()["caption"]
            except requests.exceptions.RequestException as e:
                print(f"Error calling Qwen-VL API: {str(e)}")
                this_caption = "Error generating caption"
                
            caption_result[this_segment] = f"Caption:\n{this_caption}\nTranscript:\n{segment_transcript}\n\n"
            
        return caption_result
    except Exception as e:
        print(f"Error in retrieved_segment_caption: {str(e)}")
        return {}