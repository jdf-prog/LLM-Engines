import time
import os
import os
import json
import hashlib
from pathlib import Path
from typing import Union, List
from typing import List
from cachetools import LRUCache
from functools import lru_cache
from .cache_utils import get_cache_file
from tqdm import tqdm
from collections import defaultdict

# Global cache dictionary using LRUCache
cache_dict = {}
loaded_cache_files = defaultdict(list)
# Adjust this value based on your memory constraints and requirements
MAX_CACHE_SIZE = 500000  # Example: 100k items

class LRUCacheManager:
    def __init__(self, maxsize=MAX_CACHE_SIZE):
        self.cache = LRUCache(maxsize=maxsize)

    def get(self, key):
        if key not in self.cache:
            return None
        return self.cache.get(key)

    def set(self, key, value):
        self.cache[key] = value

    def __getitem__(self, key):
        return self.get(key)
    
    def __setitem__(self, key, value):
        self.set(key, value)
    
    def __del__(self):
        self.save()

def load_cache(model_name, cache_dir=None):
    global cache_dict
    global loaded_cache_files
    
    if model_name not in cache_dict:
        if cache_dir is None:
            cache_dir = Path(os.path.expanduser(f"~/llm_engines/generation_cache"))
        else:
            cache_dir = Path(cache_dir)
        cache_dict[model_name] = LRUCacheManager()
        cache_file = get_cache_file(model_name, cache_dir)
        if cache_file.exists():
            print("Cache file exists at:", cache_file.absolute())
            if model_name not in loaded_cache_files or cache_file.absolute() not in loaded_cache_files[model_name]:
                with open(cache_file, "r") as f:
                    for line in tqdm(f, desc="Loading cache for model: " + model_name):
                        item = json.loads(line)
                        key = list(item.keys())[0]
                        value = list(item.values())[0]
                        # cache_dict[model_name][key] = {"output": value["output"]}
                        cache_dict[model_name][key] = value
                loaded_cache_files[model_name].append(cache_file.absolute())
        
    return cache_dict[model_name]

