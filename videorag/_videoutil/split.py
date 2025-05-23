import os
import time
import shutil
import numpy as np
from tqdm import tqdm
from moviepy.video import fx as vfx
from moviepy.video.io.VideoFileClip import VideoFileClip
import multiprocessing
from functools import partial
from .._utils import logger

def save_audio_segments(
    video_path,
    working_dir,
    segment_length,
    num_frames_per_segment,
    audio_output_format='mp3',
):  
    unique_timestamp = str(int(time.time() * 1000))
    video_name = os.path.basename(video_path).split('.')[0]
    video_segment_cache_path = os.path.join(working_dir, '_cache', video_name)
    if os.path.exists(video_segment_cache_path):
        shutil.rmtree(video_segment_cache_path)
    os.makedirs(video_segment_cache_path, exist_ok=False)
    
    segment_index = 0
    segment_index2name, segment_times_info = {}, {}
    with VideoFileClip(video_path) as video:
        total_video_length = int(video.duration)
        start_times = list(range(0, total_video_length, segment_length))
        # if the last segment is shorter than 5 seconds, we merged it to the last segment
        if len(start_times) > 1 and (total_video_length - start_times[-1]) < 5:
            start_times = start_times[:-1]
        
        for start in tqdm(start_times, desc=f"Saving Audio Segments {video_name}"):
            if start != start_times[-1]:
                end = min(start + segment_length, total_video_length)
            else:
                end = total_video_length
            
            subvideo = video.subclip(start, end)
            subvideo_length = subvideo.duration
            frame_times = np.linspace(0, subvideo_length, num_frames_per_segment, endpoint=False)
            frame_times += start
            
            segment_index2name[f"{segment_index}"] = f"{unique_timestamp}-{segment_index}-{start}-{end}"
            segment_times_info[f"{segment_index}"] = {"frame_times": frame_times, "timestamp": (start, end)}
            
            # save audio segment
            audio_file_base_name = segment_index2name[f"{segment_index}"]
            audio_file = f'{audio_file_base_name}.{audio_output_format}'
            subaudio = subvideo.audio
            subaudio.write_audiofile(
                os.path.join(video_segment_cache_path, audio_file),
                codec='mp3',
                verbose=False,
                logger=None
            )
            
            segment_index += 1

    return segment_index2name, segment_times_info

def save_single_video_segment(args):
    video_path, cache_path, index, segment_name, start, end, video_output_format = args
    try:
        with VideoFileClip(video_path) as video:
            subvideo = video.subclip(start, end)
            output_path = os.path.join(cache_path, f'{segment_name}.{video_output_format}')
            subvideo.write_videofile(output_path, codec='libx264', verbose=False, logger=None)
        return True
    except Exception as e:
        return f"Error processing segment {index}: {str(e)}"

def saving_video_segments(
    video_name,
    video_path,
    working_dir,
    segment_index2name,
    segment_times_info,
    error_queue,
    video_output_format='mp4',
):
    try:
        video_segment_cache_path = os.path.join(working_dir, '_cache', video_name)
        
        # Prepare arguments for parallel processing
        args_list = []
        for index in segment_index2name:
            start, end = segment_times_info[index]["timestamp"][0], segment_times_info[index]["timestamp"][1]
            args = (video_path, video_segment_cache_path, index, segment_index2name[index], start, end, video_output_format)
            args_list.append(args)
        
        # Use multiprocessing to save segments in parallel
        # With 520GB RAM, we can use more processes. Each video segment typically needs 1-2GB RAM
        # Using 32 processes should be safe and provide good performance
        num_processes = min(multiprocessing.cpu_count(), 32)  # Increased from 4 to 32 for high-memory systems
        logger.info(f"Using {num_processes} processes to save video segments")
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap(save_single_video_segment, args_list),
                total=len(args_list),
                desc=f"Saving Video Segments {video_name}"
            ))
        
        # Check for errors
        errors = [r for r in results if r is not True]
        if errors:
            error_msg = "Errors occurred while saving video segments:\n" + "\n".join(errors)
            logger.error(error_msg)
            error_queue.put(error_msg)
            raise RuntimeError(error_msg)
            
        # Verify all segments were saved
        saved_segments = set()
        for file in os.listdir(video_segment_cache_path):
            if file.endswith(f'.{video_output_format}'):
                try:
                    parts = file.split('.')[0].split('-')
                    if len(parts) >= 4:
                        segment_index = int(parts[1])
                        saved_segments.add(segment_index)
                except (IndexError, ValueError):
                    continue
        
        expected_segments = set(int(idx) for idx in segment_index2name.keys())
        if expected_segments != saved_segments:
            error_msg = f"Failed to save all video segments. Expected {len(expected_segments)} segments, found {len(saved_segments)} segments."
            logger.error(error_msg)
            error_queue.put(error_msg)
            raise RuntimeError(error_msg)
            
    except Exception as e:
        error_msg = f"Error in saving_video_segments:\n {str(e)}"
        logger.error(error_msg)
        error_queue.put(error_msg)
        raise RuntimeError(error_msg)