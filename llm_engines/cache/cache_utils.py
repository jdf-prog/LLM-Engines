from typing import List, Union
from cachetools import LRUCache
from functools import lru_cache
import os
import hashlib
from pathlib import Path

@lru_cache(maxsize=None)
def get_cache_file(model_name, cache_dir):
    if cache_dir is not None:
        return Path(cache_dir) / f"{model_name}.jsonl"
    return Path(os.path.expanduser(f"~/llm_engines/generation_cache/{model_name}.jsonl"))

def get_inputs_hash(inputs, conv_system_msg):
    if isinstance(inputs, str):
        return hashlib.md5(inputs.encode()).hexdigest()
    if conv_system_msg:
        return hashlib.md5((conv_system_msg + "".join(inputs)).encode()).hexdigest()
    return hashlib.md5("".join(inputs).encode()).hexdigest()