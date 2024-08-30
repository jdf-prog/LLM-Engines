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

# Global cache dictionary using LRUCache
cache_dict = {}
# Adjust this value based on your memory constraints and requirements
MAX_CACHE_SIZE = 100000  # Example: 100k items

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
    if model_name not in cache_dict:
        cache_dict[model_name] = LRUCacheManager()
        cache_file = get_cache_file(model_name, cache_dir)
        if cache_file.exists():
            print("Cache file exists at:", cache_file.absolute())
            print(f"Loading cache from {cache_file}")
            with open(cache_file, "r") as f:
                for line in f:
                    item = json.loads(line)
                    key = list(item.keys())[0]
                    value = list(item.values())[0]
                    # cache_dict[model_name][key] = {"output": value["output"]}
                    cache_dict[model_name][key] = value
    return cache_dict[model_name]

