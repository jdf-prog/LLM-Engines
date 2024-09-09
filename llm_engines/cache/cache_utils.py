from typing import List, Union
from cachetools import LRUCache
from functools import lru_cache
import os
import hashlib
from pathlib import Path

@lru_cache(maxsize=None)
def get_cache_file(model_name_or_path, cache_dir):
    model_name = model_name_or_path.split("/")[-2:]
    model_name = "/".join(model_name)
    if cache_dir is not None:
        cache_file = Path(cache_dir) / f"{model_name}.jsonl"
    else:
        cache_file = Path(os.path.expanduser(f"~/llm_engines/generation_cache/{model_name}.jsonl"))
    if not cache_file.parent.exists():
        cache_file.parent.mkdir(parents=True)
    return cache_file


def get_inputs_hash(inputs, conv_system_msg):
    if isinstance(inputs, str):
        return hashlib.md5(inputs.encode()).hexdigest()
    if conv_system_msg:
        return hashlib.md5((conv_system_msg + "".join(inputs)).encode()).hexdigest()
    return hashlib.md5("".join(inputs).encode()).hexdigest()