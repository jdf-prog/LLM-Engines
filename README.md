# LLM-Engines

A unified inference engine for large language models (LLMs) including open-source models (VLLM, SGLang, Together) and commercial models (OpenAI, Mistral, Claude).

The correctness of the inference has been verified by comparing the outputs of the models with different engines when `temperature=0.0` and `max_tokens=None`.
For example, the outputs of a single model using 3 enginer (VLLM, SGLang, Together) will be the same when `temperature=0.0` and `max_tokens=None`.
Try examples below to see the outputs of different engines.

## Installation
    
```bash
pip install llm-engines
# or
pip install git+https://github.com/jdf-prog/LLM-Engines.git
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
response = call_worker_func(["What is the capital of France?"], temperature=0.0, max_tokens=None)
print(response)
```

- use together
```python
# export TOGETHER_API_KEY="your_together_api_key"
call_worker_func = get_call_worker_func(
    model_name="meta-llama/Llama-3-8b-chat-hf", 
    engine="together",
    use_cache=False
)
response = call_worker_func(["What is the capital of France?"], temperature=0.0, max_tokens=None)
print(response)
```

- openai models
```python
# export OPENAI_API_KEY="your_openai_api_key"
from llm_engines import get_call_worker_func
call_worker_func = get_call_worker_func(
    model_name="gpt-3.5-turbo", 
    engine="openai", # or one of "vllm", "together", "openai", "mistral", "claude",
    use_cache=False
)
response = call_worker_func(["What is the capital of France?"], temperature=0.0, max_tokens=None)
print(response)
```

- mistral models
```python
# export MISTRAL_API_KEY="your_mistral_api_key"
from llm_engines import get_call_worker_func
call_worker_func = get_call_worker_func(
    model_name="mistral-large-latest", 
    engine="mistral", # or one of "vllm", "together", "openai", "mistral", "claude",
    use_cache=False
)
response = call_worker_func(["What is the capital of France?"], temperature=0.0, max_tokens=None)
print(response)
```

- claude models
```python
# export ANTHROPIC_API_KEY="your_claude_api_key"
from llm_engines import get_call_worker_func
call_worker_func = get_call_worker_func(
    model_name="claude-3-opus-20240229", 
    engine="claude", # or one of "vllm", "together", "openai", "mistral", "claude",
    use_cache=False
)
response = call_worker_func(["What is the capital of France?"], temperature=0.0, max_tokens=None)
print(response)
```

- gemini models
```python
# export GOOGLE_API_KEY="your_gemini_api_key"
from llm_engines import get_call_worker_func
call_worker_func = get_call_worker_func(
    model_name="gemini-1.5-flash", 
    engine="gemini", # or one of "vllm", "together", "openai", "mistral", "claude",
    use_cache=False
)
response = call_worker_func(["What is the capital of France?"], temperature=0.0, max_tokens=None)
print(response)
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

### Parallel infernece
Check out [`./examples/mp_inference_wildchat.py`](./examples/mp_inference_wildchat.py) for parallel inference with multiple models.
```bash
python examples/mp_inference_wildchat.py
```

### Cache
all the queries and responses are cached in the `generation_cache` folder, no duplicate queries will be sent to the model.
The cache of each model is saved to `generation_cache/{model_name}.jsonl`

Example items in the cache:
```json
{"cb0b4aaf80c43c9973aefeda1bd72890": {"input": ["What is the capital of France?"], "output": "The capital of France is Paris."}}
```
The hash key here is the hash of the concatenated inputs.

### Chat template
For each open-source models, we use the default chat template as follows:
```python
prompt = self.tokenizer.apply_chat_template(
    messages, 
    add_generation_prompt=add_generation_prompt,
    tokenize=False,
    chat_template=chat_template,
)
```
There will be errors if the model does not support the chat template. 


### Launch a separate vllm/sglang model worker

- launch a separate vllm worker

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --dtype auto --host "127.0.0.1" --port 34200 --tensor-parallel-size 1 --disable-log-requests &
# address: http://127.0.0.1:34200
```

- launch a separate sglang worker
```bash
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --dtype auto --host "127.0.0.1" --port 34201 --tp-size 1 &
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --dtype auto --host "127.0.0.1" --port 34201 --tp-size 1 --disable-flashinfer & # disable flashinfer if it's not installed
# address: http://127.0.0.1:34201
```

- query multiple workers
```python
from llm_engines import get_call_worker_func
call_worker_func = get_call_worker_func(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct", 
    worker_addrs=["http://127.0.0.1:34200", "http://127.0.0.1:34201"], # many workers can be used, will be load balanced
    engine="sglang", 
    use_cache=False
)
response = call_worker_func(["What is the capital of France?"], temperature=0.0, max_tokens=None)
print(response)
# The capital of France is Paris.
```

### Test notes

When setting `temperature=0.0` and `max_tokens=None`, testing long generations:
- VLLM (fp16) can generate same outputs with hugging face transformers (fp16) generations, but not for bf16.
- Together AI can generate almost the same outputs with vllm (fp16, bf16) generations
- SGLang's outputs outputs not consistent with others.
- note that some weird inputs will cause the models to inference forever, it's better to set `timeout=30` to drop the request after certain seconds.
