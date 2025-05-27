import asyncio
import os
import torch
from dataclasses import dataclass
import numpy as np
from nano_vectordb import NanoVectorDB
from tqdm import tqdm
from imagebind.models import imagebind_model

from .._utils import logger
from ..base import BaseVectorStorage
from .._videoutil import encode_video_segments, encode_string_query


@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = 0.2
    
    def __post_init__(self):

        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["llm"]["embedding_batch_num"]
        self._client = NanoVectorDB(
            self.embedding_func.embedding_dim, storage_file=self._client_file_name
        )
        self.cosine_better_than_threshold = self.global_config.get(
            "query_better_than_threshold", self.cosine_better_than_threshold
        )

    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        embeddings = np.concatenate(embeddings_list)
        for i, d in enumerate(list_data):
            d["__vector__"] = embeddings[i]
        results = self._client.upsert(datas=list_data)
        return results

    async def query(self, query: str, top_k=5):
        embedding = await self.embedding_func([query])
        embedding = embedding[0]
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results

    async def index_done_callback(self):
        self._client.save()


@dataclass
class NanoVectorDBVideoSegmentStorage(BaseVectorStorage):
    embedding_func = None
    segment_retrieval_top_k: float = 2
    
    def __post_init__(self):
        
        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["video_embedding_batch_num"]
        self._client = NanoVectorDB(
            self.global_config["video_embedding_dim"], storage_file=self._client_file_name
        )
        self.top_k = self.global_config.get(
            "segment_retrieval_top_k", self.segment_retrieval_top_k
        )
    
    async def upsert(self, video_name, segment_index2name, video_output_format):
        # Initialize multiple GPU workers
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            logger.warning("No GPU available, falling back to CPU")
            embedders = [imagebind_model.imagebind_huge(pretrained=True)]
        else:
            logger.info(f"Using {num_gpus} GPUs for parallel processing")
            embedders = [imagebind_model.imagebind_huge(pretrained=True).cuda(i) for i in range(num_gpus)]
        
        # Prepare data
        list_data = []
        video_segment_cache_path = os.path.join(self.working_dir, '_cache', video_name)
        for index, segment_name in segment_index2name.items():
            segment_path = os.path.join(video_segment_cache_path, f"{segment_name}.{video_output_format}")
            if os.path.exists(segment_path):
                list_data.append({
                    "__id__": f"{video_name}_{index}",
                    "video_name": video_name,
                    "segment_index": index,
                    "segment_name": segment_name,
                    "segment_path": segment_path,
                })
        
        # Split data into batches for parallel processing
        batch_size = 8  # Adjust based on your GPU memory
        batches = [list_data[i:i + batch_size] for i in range(0, len(list_data), batch_size)]
        
        # Process batches in parallel using multiple GPUs
        embeddings = []
        for batch_idx, _batch in enumerate(tqdm(batches, desc=f"Encoding Video Segments {video_name}")):
            # Select GPU worker for this batch
            gpu_idx = batch_idx % num_gpus
            embedder = embedders[gpu_idx]
            
            # Get video paths for this batch
            batch_paths = [d["segment_path"] for d in _batch]
            
            # Encode batch
            batch_embeddings = encode_video_segments(batch_paths, embedder)
            embeddings.append(batch_embeddings)
        
        # Combine all embeddings
        embeddings = torch.concat(embeddings, dim=0)
        embeddings = embeddings.numpy()
        
        # Add embeddings to data
        for i, d in enumerate(list_data):
            d["__vector__"] = embeddings[i]
        
        # Save to database
        results = self._client.upsert(datas=list_data)
        return results
    
    async def query(self, query: str):
        embedder = imagebind_model.imagebind_huge(pretrained=True).cuda()
        embedder.eval()
        
        embedding = encode_string_query(query, embedder)
        embedding = embedding[0]
        results = self._client.query(
            query=embedding,
            top_k=self.top_k,
            better_than_threshold=-1,
        )
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results
    
    async def index_done_callback(self):
        self._client.save()
