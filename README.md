# LLM-Engines

A unified inference engine for large language models (LLMs) including open-source models (VLLM, SGLang, Together) and commercial models (OpenAI, Mistral, Claude).

The correctness of the inference has been verified by comparing the outputs of the models with different engines when `temperature=0.0` and `max_tokens=None`.
For example, the outputs of a single model using 3 enginer (VLLM, SGLang, Together) will be the same when `temperature=0.0` and `max_tokens=None`.
Try examples below to see the outputs of different engines.

## Installation
    
```bash
pip install git+https://github.com/jdf-progLLM-Engines.git
# pip install -e . # for development
```

## Usage

### Engines
- use sglang or vllm
```python
from llm_engines import get_call_worker_func
call_worker_func = get_call_worker_func(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct", 
    num_workers=1, # number of workers
    num_gpu_per_worker=1, # tensor parallelism size for each worker
    engine="sglang", # or "vllm"
    use_cache=False
)
call_worker_func(["What is the capital of France?"], temperature=0.0, max_tokens=None)
```

- use together
```python
call_worker_func = get_call_worker_func(
    model_name="meta-llama/Llama-3-8b-chat-hf", 
    engine="together",
    use_cache=False
)
call_worker_func(["What is the capital of France?"], temperature=0.0, max_tokens=None)

- openai models
```python
from llm_engines import get_call_worker_func
call_worker_func = get_call_worker_func(
    model_name="gpt-3.5-turbo", 
    num_workers=1, # number of workers
    num_gpu_per_worker=1, # tensor parallelism size for each worker
    engine="openai", # or one of "vllm", "together", "openai", "mistral", "claude",
    use_cache=False
)
call_worker_func(["What is the capital of France?"], temperature=0.0, max_tokens=None)
```

- mistral models
```python
from llm_engines import get_call_worker_func
call_worker_func = get_call_worker_func(
    model_name="mistral-large-latest", 
    engine="mistral", # or one of "vllm", "together", "openai", "mistral", "claude",
    use_cache=False
)
```

- claude models
```python
from llm_engines import get_call_worker_func
call_worker_func = get_call_worker_func(
    model_name="claude-3-opus-20240229", 
    engine="claude", # or one of "vllm", "together", "openai", "mistral", "claude",
    use_cache=False
)
```
### Multi-turn conversation
```python
from llm_engines import get_call_worker_func
call_worker_func = get_call_worker_func(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct", 
    num_workers=1, # number of workers
    num_gpu_per_worker=1, # tensor parallelism size for each worker
    engine="sglang", # or "vllm"
    use_cache=False
)
messages = [
    "Hello", # user message 
    "Hello! It's nice to meet you. Is there something I can help you with, or would you like to chat?", # previous model response
    "What is the capital of France?" # user message
]
call_worker_func(messages, temperature=0.0, max_tokens=None)
```
the messages should be in the format of `[user_message, model_response, user_message, model_response, ...]`

## Cache
all the queries and responses are cached in the `generation_cache` folder, no duplicate queries will be sent to the model.

Example items in the cache:
```json
{"cb0b4aaf80c43c9973aefeda1bd72890": {"input": ["What is the capital of France?"], "output": "The capital of France is Paris."}}
```
The hash key here is the hash of the concatenated inputs.