import os
import torch
import logging
from tqdm import tqdm
from faster_whisper import WhisperModel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from .._utils import logger

def speech_to_text(video_name, working_dir, segment_index2name, audio_output_format):
    model = WhisperModel("./faster-distil-whisper-large-v3")
    model.logger.setLevel(logging.WARNING)
    
    cache_path = os.path.join(working_dir, '_cache', video_name)
    logger.info(f"Debug - Starting speech recognition for {video_name}")
    logger.info(f"Debug - Number of segments to process: {len(segment_index2name)}")
    
    transcripts = {}
    for index in tqdm(segment_index2name, desc=f"Speech Recognition {video_name}"):
        segment_name = segment_index2name[index]
        audio_file = os.path.join(cache_path, f"{segment_name}.{audio_output_format}")
        # logger.info(f"Debug - Processing segment {index}, audio file: {audio_file}")
        
        if not os.path.exists(audio_file):
            logger.error(f"Debug - Audio file not found: {audio_file}")
            continue
            
        try:
            segments, info = model.transcribe(audio_file)
            result = ""
            for segment in segments:
                result += "[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment.text)
            transcripts[index] = result
            # logger.info(f"Debug - Successfully transcribed segment {index}")
        except Exception as e:
            logger.error(f"Debug - Error transcribing segment {index}: {str(e)}")
            continue
    
    logger.info(f"Debug - Completed speech recognition. Generated {len(transcripts)} transcripts")
    return transcripts