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

@lru_cache(maxsize=None)
def get_batch_cache_dir(model_name_or_path, cache_dir):
    model_name = model_name_or_path.split("/")[-2:]
    model_name = "/".join(model_name)
    if cache_dir is not None:
        batch_cache_dir = Path(cache_dir) / f"{model_name}_batch_cache"
    else:
        batch_cache_dir = Path(os.path.expanduser(f"~/llm_engines/generation_cache/{model_name}_batch_cache"))
    if not batch_cache_dir.exists():
        batch_cache_dir.mkdir(parents=True)
    return batch_cache_dir


def get_inputs_hash(inputs, conv_system_msg, generate_kwargs=None):
    
    inputs = inputs.copy()
    if isinstance(inputs, str):
        inputs = [inputs]
    
    if conv_system_msg:
        to_hash_inputs = [conv_system_msg] + inputs
    else:
        to_hash_inputs = inputs
    
    if generate_kwargs:
        to_hash_inputs.append(str(generate_kwargs))
    
    try:
        return hashlib.md5("".join(to_hash_inputs).encode()).hexdigest()
    except UnicodeEncodeError as e:
        return hashlib.md5("".join(to_hash_inputs).encode('utf-16', 'surrogatepass').decode('utf-16').encode('utf-8')).hexdigest()