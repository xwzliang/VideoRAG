import os
import sys
import json
import shutil
import asyncio
import multiprocessing
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Callable, Dict, List, Optional, Type, Union, cast
from transformers import AutoModel, AutoTokenizer
import tiktoken
import requests
from moviepy.editor import VideoFileClip
import time
import numpy as np
from tqdm import tqdm
from pydub import AudioSegment


from ._llm import (
    LLMConfig,
    openai_config,
    azure_openai_config,
    ollama_config
)
from ._op import (
    chunking_by_video_segments,
    extract_entities,
    get_chunks,
    videorag_query,
    videorag_query_multiple_choice,
)
from ._storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NanoVectorDBVideoSegmentStorage,
    NetworkXStorage,
)
from ._utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    wrap_embedding_func_with_attrs,
    convert_response_to_json,
    always_get_an_event_loop,
    logger,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)
from ._videoutil import(
    save_audio_segments,
    speech_to_text,
    segment_caption,
    merge_segment_information,
    saving_video_segments,
)
from ._videoutil.caption import load_vision_model, unload_vision_model, load_llm_model, unload_llm_model


@dataclass
class VideoRAG:
    working_dir: str = field(
        default_factory=lambda: f"./videorag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    
    # video
    threads_for_split: int = 10
    video_segment_length: int = 30 # seconds
    rough_num_frames_per_segment: int = 5 # frames
    fine_num_frames_per_segment: int = 15 # frames
    video_output_format: str = "mp4"
    audio_output_format: str = "mp3"
    video_embedding_batch_num: int = 2
    segment_retrieval_top_k: int = 4
    video_embedding_dim: int = 1024
    
    # query
    retrieval_topk_chunks: int = 2
    query_better_than_threshold: float = 0.2
    
    # graph mode
    enable_local: bool = True
    enable_naive_rag: bool = True

    # text chunking
    chunk_func: Callable[
        [
            list[list[int]],
            List[str],
            tiktoken.Encoding,
            Optional[int],
        ],
        List[Dict[str, Union[str, int]]],
    ] = chunking_by_video_segments
    chunk_token_size: int = 4096
    # chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o"

    # entity extraction
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500

    # Change to your LLM provider
    llm: LLMConfig = field(default_factory=openai_config)
    
    # entity extraction
    entity_extraction_func: callable = extract_entities
    
    # storage
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    vs_vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBVideoSegmentStorage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage
    enable_llm_cache: bool = True

    # extension
    always_create_working_dir: bool = True
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    def __post_init__(self):
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"VideoRAG init with param:\n\n  {_print_config}\n")
        
        if not os.path.exists(self.working_dir) and self.always_create_working_dir:
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        # Initialize storage instances
        self.video_path_db = self.key_string_value_json_storage_cls(
            namespace="video_path", global_config=asdict(self)
        )
        
        self.video_segments = self.key_string_value_json_storage_cls(
            namespace="video_segments", global_config=asdict(self)
        )

        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=asdict(self)
        )

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=asdict(self)
            )
            if self.enable_llm_cache
            else None
        )

        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation", global_config=asdict(self)
        )

        self.embedding_func = limit_async_func_call(self.llm.embedding_func_max_async)(wrap_embedding_func_with_attrs(
                embedding_dim = self.llm.embedding_dim,
                max_token_size = self.llm.embedding_max_token_size,
                model_name = self.llm.embedding_model_name)(self.llm.embedding_func))
        self.entities_vdb = (
            self.vector_db_storage_cls(
                namespace="entities",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
                meta_fields={"entity_name"},
            )
            if self.enable_local
            else None
        )
        self.chunks_vdb = (
            self.vector_db_storage_cls(
                namespace="chunks",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
            )
            if self.enable_naive_rag
            else None
        )
        
        self.video_segment_feature_vdb = (
            self.vs_vector_db_storage_cls(
                namespace="video_segment_feature",
                global_config=asdict(self),
                embedding_func=None, # we code the embedding process inside the insert() function.
            )
        )
        
        self.llm.best_model_func = limit_async_func_call(self.llm.best_model_max_async)(
            partial(self.llm.best_model_func, hashing_kv=self.llm_response_cache)
        )
        self.llm.cheap_model_func = limit_async_func_call(self.llm.cheap_model_max_async)(
            partial(self.llm.cheap_model_func, hashing_kv=self.llm_response_cache)
        )

        # Initialize caption model and tokenizer attributes
        self.caption_model = None
        self.caption_tokenizer = None

        # Load cached data if it exists
        self._load_cached_data()

    def _load_cached_data(self):
        """Load cached data from storage if it exists."""
        try:
            # Load video path data
            if hasattr(self.video_path_db, '_data') and self.video_path_db._data:
                logger.info("Loading cached video path data...")
                if hasattr(self.video_path_db, 'load'):
                    self.video_path_db.load()

            # Load video segments data
            if hasattr(self.video_segments, '_data') and self.video_segments._data:
                logger.info("Loading cached video segments data...")
                if hasattr(self.video_segments, 'load'):
                    self.video_segments.load()

            # Load text chunks data
            if hasattr(self.text_chunks, '_data') and self.text_chunks._data:
                logger.info("Loading cached text chunks data...")
                if hasattr(self.text_chunks, 'load'):
                    self.text_chunks.load()

            # Load LLM response cache if enabled
            if self.enable_llm_cache and hasattr(self.llm_response_cache, '_data') and self.llm_response_cache._data:
                logger.info("Loading cached LLM responses...")
                if hasattr(self.llm_response_cache, 'load'):
                    self.llm_response_cache.load()

            # Load entity graph data
            if hasattr(self.chunk_entity_relation_graph, '_graph') and self.chunk_entity_relation_graph._graph:
                logger.info("Loading cached entity graph data...")
                if hasattr(self.chunk_entity_relation_graph, 'load'):
                    self.chunk_entity_relation_graph.load()

            # Load vector database data
            if self.enable_local and hasattr(self.entities_vdb, 'load'):
                logger.info("Loading cached entity vector database...")
                self.entities_vdb.load()

            if self.enable_naive_rag and hasattr(self.chunks_vdb, 'load'):
                logger.info("Loading cached chunks vector database...")
                self.chunks_vdb.load()

            if hasattr(self.video_segment_feature_vdb, 'load'):
                logger.info("Loading cached video segment features...")
                self.video_segment_feature_vdb.load()

            logger.info("Successfully loaded all cached data")
        except Exception as e:
            logger.error(f"Error loading cached data: {str(e)}")
            logger.info("Continuing without cached data...")

    def load_caption_model(self, debug=False):
        # caption model
        if not debug:
            import requests
            try:
                # Request server to load the vision model
                response = requests.post("http://localhost:8000/load_model")
                response.raise_for_status()
                self.caption_model = "Qwen/Qwen-VL-Chat"  # This will be used via REST API
                self.caption_tokenizer = "Qwen/Qwen-VL-Chat"  # Set this to match model name for compatibility
            except requests.exceptions.RequestException as e:
                print(f"Error loading Qwen-VL model: {str(e)}")
                raise RuntimeError("Failed to load Qwen-VL model")
        else:
            self.caption_model = None
            self.caption_tokenizer = None
    
    def insert_video(self, video_path_list=None):
        loop = always_get_an_event_loop()
        for video_path in video_path_list:
            # Step0: check the existence
            video_name = os.path.basename(video_path).split('.')[0]
            if video_name in self.video_segments._data:
                logger.info(f"Find the video named {os.path.basename(video_path)} in storage.")
                # Check if we have all necessary data (transcripts and captions)
                current_data = self.video_segments._data.get(video_name, {})
                has_all_data = True
                for index in current_data:
                    if "transcript" not in current_data[index] or "content" not in current_data[index]:
                        has_all_data = False
                        break
                
                if has_all_data:
                    logger.info(f"Video {video_name} has all necessary data (transcripts and captions), skipping processing.")
                    continue
                else:
                    logger.info(f"Video {video_name} found but missing some data, proceeding with processing...")
                    # If we have transcripts but missing captions, use the cached transcripts
                    if all("transcript" in current_data[index] for index in current_data):
                        logger.info(f"Using cached transcripts for {video_name}")
                        transcripts = {index: current_data[index]["transcript"] for index in current_data}
                    else:
                        transcripts = None
            else:
                transcripts = None
                
            loop.run_until_complete(self.video_path_db.upsert(
                {video_name: video_path}
            ))
            
            # Check if video segments already exist in the working directory
            video_segment_cache_path = os.path.join(self.working_dir, '_cache', video_name)
            if os.path.exists(video_segment_cache_path):
                logger.info(f"Found existing video segments for {video_name}, loading from cache...")
                # Load existing segment information
                segment_index2name = {}
                segment_times_info = {}
                has_video_segments = False
                has_audio_segments = False
                processed_segments = set()  # Track all processed segment indices
                
                for segment_file in os.listdir(video_segment_cache_path):
                    try:
                        # Try different filename formats
                        if segment_file.endswith(f'.{self.video_output_format}'):
                            has_video_segments = True
                            # Try to extract segment index from filename
                            # Format: timestamp-segment-start-end.mp4
                            try:
                                # Extract the segment number (second number in the filename)
                                parts = segment_file.split('.')[0].split('-')
                                if len(parts) >= 4:  # timestamp-segment-start-end
                                    segment_index = int(parts[1])
                                    start_time = int(parts[2])
                                    end_time = int(parts[3])
                                    # Store segment name without extension
                                    segment_index2name[segment_index] = os.path.splitext(segment_file)[0]
                                    # Calculate frame times for the segment
                                    frame_times = np.linspace(start_time, end_time, self.rough_num_frames_per_segment, endpoint=False)
                                    segment_times_info[segment_index] = {
                                        'timestamp': [start_time, end_time],
                                        'duration': end_time - start_time,
                                        'frame_times': frame_times
                                    }
                                    processed_segments.add(segment_index)
                                else:
                                    logger.warning(f"Unexpected filename format: {segment_file}")
                                    continue
                            except (IndexError, ValueError) as e:
                                logger.warning(f"Could not parse segment index from filename: {segment_file}, error: {str(e)}")
                                continue
                            
                        elif segment_file.endswith(f'.{self.audio_output_format}'):
                            has_audio_segments = True
                            # Try to extract segment index from filename
                            # Format: timestamp-segment-start-end.mp3
                            try:
                                # Extract the segment number (second number in the filename)
                                parts = segment_file.split('.')[0].split('-')
                                if len(parts) >= 4:  # timestamp-segment-start-end
                                    segment_index = int(parts[1])
                                    start_time = int(parts[2])
                                    end_time = int(parts[3])
                                    # Add to segment_index2name for speech recognition
                                    segment_index2name[segment_index] = os.path.splitext(os.path.basename(segment_file))[0]
                                    if segment_index not in segment_times_info:
                                        # Calculate frame times for the segment
                                        frame_times = np.linspace(start_time, end_time, self.rough_num_frames_per_segment, endpoint=False)
                                        segment_times_info[segment_index] = {
                                            'timestamp': [start_time, end_time],
                                            'duration': end_time - start_time,
                                            'frame_times': frame_times
                                        }
                                    processed_segments.add(segment_index)
                                else:
                                    logger.warning(f"Unexpected filename format: {segment_file}")
                                    continue
                            except (IndexError, ValueError) as e:
                                logger.warning(f"Could not parse segment index from filename: {segment_file}, error: {str(e)}")
                                continue
                    except Exception as e:
                        logger.warning(f"Error processing file {segment_file}: {str(e)}")
                        continue
                
                if not (has_video_segments or has_audio_segments):
                    logger.info(f"Debug - has_audio_segments: {has_audio_segments}")
                    logger.info(f"Debug - has_video_segments: {has_video_segments}")
                    logger.info(f"No valid segments found in cache for {video_name}, proceeding with video split...")
                    segment_index2name, segment_times_info = save_audio_segments(
                        video_path, 
                        self.working_dir, 
                        self.video_segment_length,
                        self.rough_num_frames_per_segment,
                        self.audio_output_format
                    )
                else:
                    logger.info(f"Found {len(processed_segments)} segments in cache for {video_name} ({len(segment_index2name)} video segments, {len(processed_segments) - len(segment_index2name)} audio segments)")
                    # No need to call save_audio_segments here since we already have the segments
                    # The segment_index2name and segment_times_info are already populated from the cache
            else:
                logger.info(f"No existing segments found for {video_name}, splitting video...")
                segment_index2name, segment_times_info = save_audio_segments(
                    video_path, 
                    self.working_dir, 
                    self.video_segment_length,
                    self.rough_num_frames_per_segment,
                    self.audio_output_format
                )
            
            # Save split information
            loop.run_until_complete(self.video_segments.upsert(
                {video_name: {index: {"time": f"{info['timestamp'][0]}-{info['timestamp'][1]}"} for index, info in segment_times_info.items()}}
            ))
            
            # Step2: obtain transcript with whisper
            if transcripts is None:
                logger.info(f"No cached transcripts found for {video_name}, running speech recognition...")
                transcripts = speech_to_text(
                    video_name, 
                    self.working_dir, 
                    segment_index2name,
                    self.audio_output_format
                )
                # Save transcripts immediately after speech recognition
                current_data = self.video_segments._data.get(video_name, {})
                logger.info(f"Debug - Number of transcripts generated: {len(transcripts)}")
                logger.info(f"Debug - Number of segments in current_data: {len(current_data)}")
                
                for index, transcript in transcripts.items():
                    if index in current_data:
                        current_data[index]["transcript"] = transcript
                        logger.info(f"Debug - Saved transcript for segment {index}")
                    else:
                        logger.warning(f"Debug - Segment {index} not found in current_data")
                
                # Sort segments by index before saving
                sorted_data = {}
                for index in sorted(current_data.keys(), key=int):
                    sorted_data[index] = current_data[index]
                
                # Save to memory and disk
                logger.info(f"Debug - Saving {len(sorted_data)} segments with transcripts")
                loop.run_until_complete(self.video_segments.upsert({video_name: sorted_data}))
                loop.run_until_complete(self.video_segments.index_done_callback())
                
                # Verify the save was successful by checking memory
                saved_data = self.video_segments._data.get(video_name, {})
                logger.info(f"Debug - Number of segments in saved_data: {len(saved_data)}")
                missing_transcripts = [idx for idx in segment_index2name if idx not in saved_data or "transcript" not in saved_data[idx]]
                if missing_transcripts:
                    logger.warning(f"Debug - Missing transcripts for segments: {missing_transcripts}")
                    logger.warning("Failed to save transcripts, retrying...")
                    loop.run_until_complete(self.video_segments.upsert({video_name: sorted_data}))
                    loop.run_until_complete(self.video_segments.index_done_callback())
            else:
                logger.info(f"Using cached transcripts for {video_name}")
            
            # Step3: saving video segments **as well as** obtain caption with vision language model
            manager = multiprocessing.Manager()
            captions = manager.dict()
            error_queue = manager.Queue()
            
            # Check for cached captions
            caption_storage = JsonKVStorage(namespace="video_captions", global_config={"working_dir": self.working_dir})
            cached_captions = caption_storage._data.get(video_name, {})
            
            if cached_captions:
                logger.info(f"Found {len(cached_captions)} cached captions for {video_name}")
                # Add cached captions to the result
                for index, caption in cached_captions.items():
                    captions[int(index)] = caption
                logger.info(f"Loaded {len(captions)} cached captions")
            
            # First, ensure all video segments are properly saved
            video_segment_cache_path = os.path.join(self.working_dir, '_cache', video_name)
            if os.path.exists(video_segment_cache_path):
                # Check if we have all expected segments
                expected_segments = set(segment_index2name.keys())
                actual_video_segments = set()
                actual_audio_segments = set()
                
                for file in os.listdir(video_segment_cache_path):
                    try:
                        parts = file.split('.')[0].split('-')
                        if len(parts) >= 4:
                            segment_index = int(parts[1])
                            if file.endswith(f'.{self.video_output_format}'):
                                actual_video_segments.add(segment_index)
                            elif file.endswith(f'.{self.audio_output_format}'):
                                actual_audio_segments.add(segment_index)
                    except (IndexError, ValueError):
                        continue
                
                logger.info(f"Debug - Expected segments: {len(expected_segments)}")
                logger.info(f"Debug - Actual video segments: {len(actual_video_segments)}")
                logger.info(f"Debug - Actual audio segments: {len(actual_audio_segments)}")
                
                # Check and regenerate audio segments if needed
                if expected_segments != actual_audio_segments:
                    logger.warning(f"Found incomplete audio segments for {video_name}. Expected {len(expected_segments)} segments, found {len(actual_audio_segments)} segments.")
                    logger.info("Regenerating audio segments...")
                    # Remove only audio files
                    for file in os.listdir(video_segment_cache_path):
                        if file.endswith(f'.{self.audio_output_format}'):
                            os.remove(os.path.join(video_segment_cache_path, file))
                    # Regenerate audio segments
                    segment_index2name, segment_times_info = save_audio_segments(
                        video_path, 
                        self.working_dir, 
                        self.video_segment_length,
                        self.rough_num_frames_per_segment,
                        self.audio_output_format
                    )
                
                # Check and regenerate video segments if needed
                if expected_segments != actual_video_segments:
                    logger.warning(f"Found incomplete video segments for {video_name}. Expected {len(expected_segments)} segments, found {len(actual_video_segments)} segments.")
                    logger.info("Regenerating video segments...")
                    # Remove only video files
                    for file in os.listdir(video_segment_cache_path):
                        if file.endswith(f'.{self.video_output_format}'):
                            os.remove(os.path.join(video_segment_cache_path, file))
                    # Regenerate video segments
                    process_saving_video_segments = multiprocessing.Process(
                        target=saving_video_segments,
                        args=(
                            video_name,
                            video_path,
                            self.working_dir,
                            segment_index2name,
                            segment_times_info,
                            error_queue,
                            self.video_output_format,
                        )
                    )
                    process_saving_video_segments.start()
                    process_saving_video_segments.join()
                    
                    # Verify video segments were saved
                    saved_video_segments = set()
                    for file in os.listdir(video_segment_cache_path):
                        if file.endswith(f'.{self.video_output_format}'):
                            try:
                                parts = file.split('.')[0].split('-')
                                if len(parts) >= 4:
                                    segment_index = int(parts[1])
                                    saved_video_segments.add(segment_index)
                            except (IndexError, ValueError):
                                continue
                    
                    if expected_segments != saved_video_segments:
                        error_msg = f"Failed to save all video segments. Expected {len(expected_segments)} segments, found {len(saved_video_segments)} segments."
                        error_queue.put(error_msg)
                        raise RuntimeError(error_msg)
            
            # Filter segments to only include those that have both files and transcripts
            valid_segments = {}
            # Create a mapping of segment indices to their sequential order
            segment_order = {str(idx): i for i, idx in enumerate(sorted(segment_index2name.keys(), key=int))}
            
            for index in segment_index2name:
                # Get the sequential index for this segment
                seq_index = str(segment_order[str(index)])
                if seq_index in transcripts:
                    valid_segments[index] = segment_index2name[index]
            
            logger.info(f"Debug - Number of valid segments (with both files and transcripts): {len(valid_segments)}")
            logger.info(f"Debug - Total segments: {len(segment_index2name)}")
            logger.info(f"Debug - Total transcripts: {len(transcripts)}")
            logger.info(f"Debug - Segment indices: {list(segment_index2name.keys())[:5]}...")
            logger.info(f"Debug - Transcript indices: {list(transcripts.keys())[:5]}...")
            logger.info(f"Debug - Segment order mapping: {dict(list(segment_order.items())[:5])}...")
            
            if not valid_segments:
                logger.error(f"No valid segments found for {video_name} (segments with both files and transcripts)")
                raise RuntimeError(f"No valid segments found for {video_name}")
            
            # Create a mapping of segment indices to their transcripts
            transcript_mapping = {}
            for index in sorted(valid_segments.keys(), key=int):
                seq_index = str(segment_order[str(index)])
                transcript_mapping[index] = transcripts[seq_index]
            
            # Only process segments that don't have cached captions, maintaining order
            segments_to_process = {idx: name for idx, name in sorted(valid_segments.items(), key=lambda x: int(x[0])) if idx not in captions}
            logger.info(f"Processing {len(segments_to_process)} segments for captioning (skipping {len(valid_segments) - len(segments_to_process)} cached segments)")
            logger.info(f"Segment order: {list(segments_to_process.keys())[:5]}...")
            
            if segments_to_process:
                # Create an event loop for the async function
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    # Run the async function
                    loop.run_until_complete(segment_caption(
                        video_name,
                        video_path,
                        segments_to_process,
                        transcript_mapping,
                        segment_times_info,
                        captions,
                        error_queue,
                        self.working_dir,
                    ))
                finally:
                    loop.close()
                
                # if raise error in this two, stop the processing
                while not error_queue.empty():
                    error_message = error_queue.get()
                    with open('error_log_videorag.txt', 'a', encoding='utf-8') as log_file:
                        log_file.write(f"Video Name:{video_name} Error processing:\n{error_message}\n\n")
                    raise RuntimeError(error_message)
            
            # Save captions as they are generated
            current_data = self.video_segments._data.get(video_name, {})
            for index, caption in captions.items():
                if index in current_data:
                    current_data[index]["content"] = caption
            loop.run_until_complete(self.video_segments.upsert({video_name: current_data}))
            
            # Step4: insert video segments information
            segments_information = merge_segment_information(
                segment_index2name,
                segment_times_info,
                transcripts,
                captions,
            )
            manager.shutdown()
            loop.run_until_complete(self.video_segments.upsert(
                {video_name: segments_information}
            ))
            
            # Unload vision model before loading imagebind
            if not unload_vision_model():
                print("Failed to unload vision model")
                return []
            try:
                # Step5: encode video segment features
                loop.run_until_complete(self.video_segment_feature_vdb.upsert(
                    video_name,
                    segment_index2name,
                    self.video_output_format,
                ))
            except Exception as e:
                logger.error(f"Error in video segment feature encoding: {str(e)}")
                logger.info("Continuing with available data...")
            
            # Step6: delete the cache file
            # video_segment_cache_path = os.path.join(self.working_dir, '_cache', video_name)
            # if os.path.exists(video_segment_cache_path):
            #     shutil.rmtree(video_segment_cache_path)
            
            # Step 7: saving current video information
            loop.run_until_complete(self._save_video_segments())
        
        try:
            loop.run_until_complete(self.ainsert(self.video_segments._data))
        except Exception as e:
            logger.error(f"Error in final ainsert: {str(e)}")
            logger.info("Continuing with available data...")

    def query(self, query: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        # Load DeepSeek model for querying
        if not load_llm_model():
            raise RuntimeError("Failed to load DeepSeek model for querying")
            
        try:
            if param.mode == "videorag":
                response = await videorag_query(
                    query,
                    self.entities_vdb,
                    self.text_chunks,
                    self.chunks_vdb,
                    self.video_path_db,
                    self.video_segments,
                    self.video_segment_feature_vdb,
                    self.chunk_entity_relation_graph,
                    self.caption_model, 
                    self.caption_tokenizer,
                    param,
                    asdict(self),
                )
            # NOTE: update here
            elif param.mode == "videorag_multiple_choice":
                response = await videorag_query_multiple_choice(
                    query,
                    self.entities_vdb,
                    self.text_chunks,
                    self.chunks_vdb,
                    self.video_path_db,
                    self.video_segments,
                    self.video_segment_feature_vdb,
                    self.chunk_entity_relation_graph,
                    self.caption_model, 
                    self.caption_tokenizer,
                    param,
                    asdict(self),
                )
            else:
                raise ValueError(f"Unknown mode {param.mode}")
            await self._query_done()
            return response
        finally:
            # Always unload the model after processing
            # unload_llm_model()
            pass

    async def ainsert(self, new_video_segment):
        await self._insert_start()
        try:
            # Load DeepSeek model for entity extraction
            # if not load_llm_model():
            #     raise RuntimeError("Failed to load DeepSeek model for entity extraction")
            
            try:
                # ---------- chunking
                inserting_chunks = get_chunks(
                    new_videos=new_video_segment,
                    chunk_func=self.chunk_func,
                    max_token_size=self.chunk_token_size,
                )
                _add_chunk_keys = await self.text_chunks.filter_keys(
                    list(inserting_chunks.keys())
                )
                inserting_chunks = {
                    k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
                }
                if not len(inserting_chunks):
                    logger.warning(f"All chunks are already in the storage")
                    return
                logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")
                if self.enable_naive_rag:
                    logger.info("Insert chunks for naive RAG")
                    await self.chunks_vdb.upsert(inserting_chunks)

                # ---------- Commenting out entity extraction
                # logger.info("[Entity Extraction]...")
                # maybe_new_kg, _, _ = await self.entity_extraction_func(
                #     inserting_chunks,
                #     knowledge_graph_inst=self.chunk_entity_relation_graph,
                #     entity_vdb=self.entities_vdb,
                #     global_config=asdict(self),
                # )
                # if maybe_new_kg is None:
                #     logger.warning("No new entities found")
                #     return
                # self.chunk_entity_relation_graph = maybe_new_kg
                # ---------- commit upsertings and indexing
                await self.text_chunks.upsert(inserting_chunks)
            finally:
                # Always unload the model after processing
                # unload_llm_model()
                pass
        finally:
            await self._insert_done()

    async def _insert_start(self):
        tasks = []
        for storage_inst in [
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_start_callback())
        await asyncio.gather(*tasks)

    async def _save_video_segments(self):
        tasks = []
        for storage_inst in [
            self.video_segment_feature_vdb,
            self.video_segments,
            self.video_path_db,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
    
    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.text_chunks,
            self.llm_response_cache,
            self.entities_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
            self.video_segment_feature_vdb,
            self.video_segments,
            self.video_path_db,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def delete_video(self, video_path):
        """Delete a specific video and all its associated data from the system."""
        # Get the video name (without extension) from the path
        video_name = os.path.basename(video_path).split('.')[0]
        loop = always_get_an_event_loop()
        
        # 1. Delete video segments from working directory
        video_segment_cache_path = os.path.join(self.working_dir, '_cache', video_name)
        if os.path.exists(video_segment_cache_path):
            shutil.rmtree(video_segment_cache_path)
            
        # 2. Delete from video_path_db
        if video_name in self.video_path_db._data:
            del self.video_path_db._data[video_name]
            loop.run_until_complete(self.video_path_db.index_done_callback())
            
        # 3. Delete from video_segments
        if video_name in self.video_segments._data:
            del self.video_segments._data[video_name]
            loop.run_until_complete(self.video_segments.index_done_callback())
            
        # 4. Delete from video_segment_feature_vdb
        if hasattr(self.video_segment_feature_vdb, 'delete'):
            loop.run_until_complete(self.video_segment_feature_vdb.delete(video_name))
        
        # 5. Delete associated chunks and entities
        # First get all chunk keys associated with this video
        chunk_keys_to_delete = []
        for chunk_key in list(self.text_chunks._data.keys()):
            if chunk_key.startswith(f"{video_name}_"):
                chunk_keys_to_delete.append(chunk_key)
                
        if chunk_keys_to_delete:
            # Delete from text_chunks
            for key in chunk_keys_to_delete:
                del self.text_chunks._data[key]
            loop.run_until_complete(self.text_chunks.index_done_callback())
            
            # Delete from chunks_vdb if enabled
            if self.enable_naive_rag and hasattr(self.chunks_vdb, 'delete'):
                loop.run_until_complete(self.chunks_vdb.delete(chunk_keys_to_delete))
                
            # Delete from entities_vdb if enabled
            if self.enable_local:
                # Get all entity keys associated with these chunks
                entity_keys_to_delete = []
                for chunk_key in chunk_keys_to_delete:
                    if chunk_key in self.chunk_entity_relation_graph._graph:
                        for entity in self.chunk_entity_relation_graph._graph[chunk_key]:
                            entity_keys_to_delete.append(entity)
                
                if entity_keys_to_delete and hasattr(self.entities_vdb, 'delete'):
                    loop.run_until_complete(self.entities_vdb.delete(entity_keys_to_delete))
                    
                # Remove edges from graph
                for chunk_key in chunk_keys_to_delete:
                    if chunk_key in self.chunk_entity_relation_graph._graph:
                        del self.chunk_entity_relation_graph._graph[chunk_key]
        
        # 6. Save changes
        loop.run_until_complete(self._save_video_segments())
        logger.info(f"Successfully deleted video {video_name} and all associated data")

    def regenerate_video(self, video_path):
        """Delete and reinsert a video to regenerate all its data.
        
        Args:
            video_path (str): Path to the video file to regenerate
        """
        logger.info(f"Regenerating video: {video_path}")
        
        # First delete the video
        self.delete_video(video_path)
        logger.info(f"Successfully deleted video: {video_path}")
        
        # Then reinsert it
        self.insert_video(video_path_list=[video_path])
        logger.info(f"Successfully reinserted video: {video_path}")
        
        return True

    def regenerate_query(self, query: str, param: QueryParam = QueryParam()):
        """Clear the cache and re-run a query.
        
        Args:
            query (str): The query to regenerate
            param (QueryParam): Query parameters
            
        Returns:
            The regenerated query response
        """
        logger.info(f"Regenerating query: {query}")
        
        # Clear the LLM response cache
        if self.llm_response_cache is not None:
            self.llm_response_cache._data = {}
            logger.info("Cleared LLM response cache")
        
        # Run the query again
        response = self.query(query, param)
        logger.info(f"Successfully regenerated query response")
        
        return response
