import json
import os
import shutil
import atexit
import sqlite3
import mmap
import threading
import struct
from .cache_utils import get_cache_file
from pathlib import Path
from typing import Union, List, Dict
from cachetools import LRUCache
from tqdm import tqdm
from collections import defaultdict

BLOCK_SIZE = 3072 * 1024  # 3MB
MAX_CACHE_SIZE = 100000  # Example: 10k items
MAX_MEMORY_BLOCKS = 32 # Example: 32 blocks
# Global cache dictionary using MultiLevelCache
cache_dict = {}
loaded_cache_files = defaultdict(list)

class BlockCache:
    def __init__(self, max_size=MAX_MEMORY_BLOCKS):
        self.cache = LRUCache(maxsize=max_size)
        self.lock = threading.Lock()

    def get(self, block_id): 
        with self.lock:
            return self.cache.get(block_id)

    def set(self, block_id, data):
        with self.lock:
            self.cache[block_id] = data

class EfficientDiskCache:
    def __init__(self, cache_dir, model_name, block_size=BLOCK_SIZE, max_memory_blocks=MAX_MEMORY_BLOCKS):
        self.cache_dir = Path(cache_dir) / f"{model_name}_disk_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.block_size = block_size
        self.index_db = self.cache_dir / "index.db"
        self.block_cache = BlockCache(max_size=max_memory_blocks)
        self.db_lock = threading.Lock()
        self.init_index_db()

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        try:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
            print(f"Cleaned up cache directory: {self.cache_dir}")
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def init_index_db(self):
        with self.db_lock:
            with sqlite3.connect(self.index_db) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS cache_index
                    (key TEXT PRIMARY KEY, block_id INTEGER, offset INTEGER, length INTEGER)
                ''')
                conn.commit()

    def get_block_file(self, block_id):
        return self.cache_dir / f"block_{block_id}.bin"

    def get(self, key):
        with self.db_lock:
            with sqlite3.connect(self.index_db) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT block_id, offset, length FROM cache_index WHERE key = ?", (key,))
                result = cursor.fetchone()
        
        if result:
            block_id, offset, length = result
            block_data = self.block_cache.get(block_id)
            if block_data is None:
                block_file = self.get_block_file(block_id)
                if block_file.exists():
                    try:
                        with open(block_file, 'rb') as f:
                            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                            block_data = mm.read()
                        self.block_cache.set(block_id, block_data)
                    except Exception as e:
                        print(f"Error reading block file: {e}")
                        return None
                else:
                    return None
            
            try:
                data = block_data[offset:offset+length]
                value_length = struct.unpack('!I', data[:4])[0]
                json_data = data[4:4+value_length].decode('utf-8')
                return json.loads(json_data)
            except (struct.error, json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"Error decoding data: {e}")
                return None
        return None

    def set(self, key, value):
        json_data = json.dumps(value).encode('utf-8')
        data_length = len(json_data)
        full_data = struct.pack('!I', data_length) + json_data

        with self.db_lock:
            with sqlite3.connect(self.index_db) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT MAX(block_id) FROM cache_index")
                result = cursor.fetchone()
                current_block_id = result[0] if result[0] is not None else 0
                
                block_file = self.get_block_file(current_block_id)
                if block_file.exists():
                    file_size = block_file.stat().st_size
                    if file_size + len(full_data) > self.block_size:
                        current_block_id += 1
                        block_file = self.get_block_file(current_block_id)
                        file_size = 0
                else:
                    file_size = 0

                with open(block_file, 'ab') as f:
                    f.write(full_data)
                    
                cursor.execute('''
                    INSERT OR REPLACE INTO cache_index (key, block_id, offset, length)
                    VALUES (?, ?, ?, ?)
                ''', (key, current_block_id, file_size, len(full_data)))
                conn.commit()

        block_data = self.block_cache.get(current_block_id)
        if block_data is not None:
            self.block_cache.set(current_block_id, block_data + full_data)

    def bulk_insert(self, data: Dict[str, Dict]):
        with self.db_lock:
            with sqlite3.connect(self.index_db) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT MAX(block_id) FROM cache_index")
                result = cursor.fetchone()
                current_block_id = result[0] if result[0] is not None else 0
                
                block_file = self.get_block_file(current_block_id)
                if block_file.exists():
                    file_size = block_file.stat().st_size
                else:
                    file_size = 0

                index_data = []
                current_block_data = b""
                
                for key, value in data.items():
                    json_data = json.dumps(value).encode('utf-8')
                    data_length = len(json_data)
                    full_data = struct.pack('!I', data_length) + json_data
                    
                    if file_size + len(full_data) > self.block_size:
                        with open(block_file, 'ab') as f:
                            f.write(current_block_data)
                        self.block_cache.set(current_block_id, current_block_data)
                        
                        current_block_id += 1
                        block_file = self.get_block_file(current_block_id)
                        file_size = 0
                        current_block_data = b""
                    
                    index_data.append((key, current_block_id, file_size, len(full_data)))
                    current_block_data += full_data
                    file_size += len(full_data)

                if current_block_data:
                    with open(block_file, 'ab') as f:
                        f.write(current_block_data)
                    self.block_cache.set(current_block_id, current_block_data)

                cursor.executemany('''
                    INSERT OR REPLACE INTO cache_index (key, block_id, offset, length)
                    VALUES (?, ?, ?, ?)
                ''', index_data)
                conn.commit()
                            
class MultiLevelCache:
    def __init__(self, model_name, cache_dir, memory_size=MAX_CACHE_SIZE):
        self.memory_cache = LRUCache(maxsize=memory_size)
        self.disk_cache = EfficientDiskCache(cache_dir, model_name)

    def get(self, key):
        value = self.memory_cache.get(key)
        if value is not None:
            return value

        value = self.disk_cache.get(key)
        if value is not None:
            self.memory_cache[key] = value
            return value

        return None
    
    def __getitem__(self, key):
        return self.get(key)

    def set(self, key, value):
        self.memory_cache[key] = value
        self.disk_cache.set(key, value)
        
    def __setitem__(self, key, value):
        self.set(key, value)

    def bulk_insert(self, data: Dict[str, Dict]):
        self.disk_cache.bulk_insert(data)
        for key, value in tqdm(data.items(), desc="Bulk inserting into Memory Cache"):
            self.memory_cache[key] = value


def load_cache(model_name, cache_dir=None):
    global cache_dict
    global loaded_cache_files
    if model_name not in cache_dict:
        if cache_dir is None:
            cache_dir = Path(os.path.expanduser(f"~/llm_engines/generation_cache"))
        else:
            cache_dir = Path(cache_dir)
        cache_file = get_cache_file(model_name, cache_dir)
        cache_dict[model_name] = MultiLevelCache(model_name, cache_dir)
        
        if cache_file.exists():
            print("Cache file exists at:", cache_file.absolute())
            if model_name not in loaded_cache_files or cache_file.absolute() not in loaded_cache_files[model_name]:
                initial_data = {}
                with open(cache_file, 'r') as f:
                    for line in tqdm(f, desc="Loading cache for model: " + model_name):
                        data = json.loads(line)
                        key = list(data.keys())[0]  
                        initial_data[key] = data[key]
                if initial_data:
                    cache_dict[model_name].bulk_insert(initial_data)
                loaded_cache_files[model_name].append(cache_file.absolute())
                
    return cache_dict[model_name]

# Cleanup function to be called at exit
def cleanup_all_caches():
    global cache_dict
    # print("Cleaning up all caches...")
    for model_name, cache in cache_dict.items():
        cache.disk_cache.cleanup()
    cache_dict.clear()

# Register the cleanup function to be called at exit
atexit.register(cleanup_all_caches)