import os
from dataclasses import dataclass

from .._utils import load_json, logger, write_json
from ..base import (
    BaseKVStorage,
)


@dataclass
class JsonKVStorage(BaseKVStorage):
    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        logger.info(f"Initializing JsonKVStorage for {self.namespace} in {working_dir}")
        
        # Ensure working directory exists
        if not os.path.exists(working_dir):
            logger.info(f"Creating working directory: {working_dir}")
            os.makedirs(working_dir)
            
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        logger.info(f"Storage file path: {self._file_name}")
        
        if os.path.exists(self._file_name):
            logger.info(f"Loading existing data from {self._file_name}")
            self._data = load_json(self._file_name) or {}
        else:
            logger.info(f"Creating new storage file at {self._file_name}")
            self._data = {}
            # Create empty file
            write_json(self._data, self._file_name)
            
        logger.info(f"Loaded KV {self.namespace} with {len(self._data)} data")

    async def all_keys(self) -> list[str]:
        return list(self._data.keys())

    async def index_done_callback(self):
        logger.info(f"Saving {len(self._data)} items to {self._file_name}")
        write_json(self._data, self._file_name)

    async def get_by_id(self, id):
        return self._data.get(id, None)

    async def get_by_ids(self, ids, fields=None):
        if fields is None:
            return [self._data.get(id, None) for id in ids]
        return [
            (
                {k: v for k, v in self._data[id].items() if k in fields}
                if self._data.get(id, None)
                else None
            )
            for id in ids
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        return set([s for s in data if s not in self._data])

    async def upsert(self, data: dict[str, dict]):
        for k, v in data.items():
            if isinstance(v, dict):
                # Only sort keys that are numeric strings
                numeric_items = []
                non_numeric_items = []
                for key, value in v.items():
                    try:
                        int(key)  # Try to convert to int
                        numeric_items.append((key, value))
                    except ValueError:
                        non_numeric_items.append((key, value))
                
                # Sort only the numeric keys
                sorted_numeric = sorted(numeric_items, key=lambda x: int(x[0]))
                # Combine sorted numeric items with non-numeric items
                self._data[k] = dict(sorted_numeric + non_numeric_items)
            else:
                self._data[k] = v

    async def drop(self):
        self._data = {}
