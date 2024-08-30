import json
import os
import shutil
import atexit
from pathlib import Path
from typing import Union, List, Dict
from cachetools import LRUCache
import sqlite3
from tqdm import tqdm
from .cache_utils import get_cache_file

BLOCK_SIZE = 1000
MAX_CACHE_SIZE = 100000  # Example: 100k items
# Global cache dictionary using MultiLevelCache
cache_dict = {}

class EfficientDiskCache:
    def __init__(self, cache_dir, model_name, block_size=BLOCK_SIZE):
        self.cache_dir = Path(cache_dir) / model_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.block_size = block_size
        self.index_db = self.cache_dir / "index.db"
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
        with sqlite3.connect(self.index_db) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cache_index
                (key TEXT PRIMARY KEY, block_id INTEGER, offset INTEGER)
            ''')
            conn.commit()

    def get_block_file(self, block_id):
        return self.cache_dir / f"block_{block_id}.json"

    def get(self, key):
        with sqlite3.connect(self.index_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT block_id, offset FROM cache_index WHERE key = ?", (key,))
            result = cursor.fetchone()
            if result:
                block_id, offset = result
                block_file = self.get_block_file(block_id)
                if block_file.exists():
                    with open(block_file, 'r') as f:
                        f.seek(offset)
                        return json.loads(f.readline().strip())
        return None

    def set(self, key, value):
        with sqlite3.connect(self.index_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(block_id) FROM cache_index")
            result = cursor.fetchone()
            current_block_id = result[0] if result[0] is not None else 0
            
            block_file = self.get_block_file(current_block_id)
            if block_file.exists():
                file_size = block_file.stat().st_size
                if file_size >= self.block_size * 1024:  # Start a new block if current one is full
                    current_block_id += 1
                    block_file = self.get_block_file(current_block_id)
                    file_size = 0
            else:
                file_size = 0

            with open(block_file, 'a') as f:
                f.seek(file_size)
                json_line = json.dumps({key: value}) + '\n'
                f.write(json_line)
                
            cursor.execute('''
                INSERT OR REPLACE INTO cache_index (key, block_id, offset)
                VALUES (?, ?, ?)
            ''', (key, current_block_id, file_size))
            conn.commit()

    def bulk_insert(self, data: Dict[str, Dict]):
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
            
            with open(block_file, 'a') as f:
                for key, value in tqdm(data.items(), desc="Bulk inserting into Disk Cache"):
                    if file_size >= self.block_size * 1024:
                        current_block_id += 1
                        block_file = self.get_block_file(current_block_id)
                        file_size = 0
                    
                    f.seek(file_size)
                    json_line = json.dumps({key: value}) + '\n'
                    f.write(json_line)
                    
                    index_data.append((key, current_block_id, file_size))
                    file_size += len(json_line)

            cursor.executemany('''
                INSERT OR REPLACE INTO cache_index (key, block_id, offset)
                VALUES (?, ?, ?)
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
    if model_name not in cache_dict:
        if cache_dir is None:
            cache_dir = Path(os.path.expanduser(f"~/llm_engines/generation_cache"))
        else:
            cache_dir = Path(cache_dir)
        cache_file = get_cache_file(model_name, cache_dir)
        cache_dict[model_name] = MultiLevelCache(model_name, cache_dir)
        
        if cache_file.exists():
            print("Cache file exists at:", cache_file.absolute())
            print(f"Loading cache for {model_name} from {cache_file}")
            initial_data = {}
            with open(cache_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    key = list(data.keys())[0]
                    initial_data[key] = data[key]
            if initial_data:
                cache_dict[model_name].bulk_insert(initial_data)
    return cache_dict[model_name]

# Cleanup function to be called at exit
def cleanup_all_caches():
    global cache_dict
    for model_name, cache in cache_dict.items():
        cache.disk_cache.cleanup()
    cache_dict.clear()

# Register the cleanup function to be called at exit
atexit.register(cleanup_all_caches)