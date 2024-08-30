from .cache_sqlite3 import load_cache # default cache
from .cache_lru import load_cache as load_lru_cache
from .cache_dict import load_cache as load_dict_cache
from .cache_utils import get_cache_file, get_inputs_hash