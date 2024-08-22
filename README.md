# LLM-Engines

A unified inference engine for large language models (LLMs) including open-source models (VLLM, SGLang, Together) and commercial models (OpenAI, Mistral, Claude).

The correctness of the inference has been verified by comparing the outputs of the models with different engines when `temperature=0.0` and `max_tokens=None`.
For example, the outputs of a single model using 3 enginer (VLLM, SGLang, Together) will be the same when `temperature=0.0` and `max_tokens=None`.
Try examples below to see the outputs of different engines.

## Installation
    
```bash
pip install llm-engines # or
# pip install git+https://github.com/jdf-prog/LLM-Engines.git

```
For development:
```bash
pip install -e . # for development
# Add ons
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/ # required for sglang
pip install flash-attn --no-build-isolation
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
    engine="vllm", # or "sglang"
    use_cache=False
)
messages = [
    "Hello", # user message 
    "Hello! It's nice to meet you. Is there something I can help you with, or would you like to chat?", # previous model response
    "What is the capital of France?" # user message
]
# or you can use opneai's multi-turn conversation format. 
messages = [
    {"role": "user", "content": "Hello"}, # user message 
    {"role": "assistant", "content": "Hello! It's nice to meet you. Is there something I can help you with, or would you like to chat?"}, # previous model response
    {"role": "user", "content": "What is the capital of France?"} # user message
]
response = call_worker_func(messages, temperature=0.0, max_tokens=None)
print(response)
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

### Worker initialization parameters
- `model_name`: the model name, e.g., "meta-llama/Meta-Llama-3-8B-Instruct" (required)
- `worker_addrs`: the list of worker addresses to use, if not provided, a new worker will be launched. If provided, it will use the existing workers (default: None)
- `num_workers`: the number of workers to use for the model (default: 1)
- `num_gpu_per_worker`: the number of GPUs to use for each worker (default: 1)
- `engine`: the engine to use, one of {vllm, sglang, together, openai, mistral, claude, gemini} (default: "vllm")
- `use_cache`: whether to use the cache for the queries and responses (default: True)
- `cache_dir`: the cache directory, (default: `~/llm-engines/generation_cache`)
- `overwrite_cache`: whether to overwrite the cache (default: False)
- `dtype`: the data type to use (default: "auto"; {auto,half,float16,bfloat16,float,float32})
- `max_retry`: the maximum number of retries for the request (default: None)
- `completion`: whether to use the completion API; If you use completion, (default: False)


### Generation parameters
- `inputs`: the list of inputs for the model; Either a list of strings or a list of dictionaries for multi-turn conversation in openai conversation format; If `completion` is True, it should be a single string (required)
- `top_p`: the nucleus sampling parameter, 0.0 means no sampling (default: 1.0)
- `temperature`: the randomness of the generation, 0.0 means deterministic generation (default: 0.0)
- `max_tokens`: the maximum number of tokens to generate, `None` means no limit (default: None)
- `timeout`: the maximum time to wait for the response, `None` means no limit (default: 300)
- `frequency_penalty`: Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim. (default: 0.0)
- `presence_penalty`: Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics. (default: 0.0)
- `n`: Number of completions to generate for each prompt. (**Do not support for llm-engines for now**) (default: 1)
- `conv_system_msg`: The system message for multi-turn conversation; If the meessage contains a system message, this parameter will be overwritten (default: None)

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
- SGLang's outputs outputs are sometimes not consistent with others.
- note that some weird inputs will cause the models to inference forever, it's better to set `timeout` (default: 300) to drop the request after certain seconds.

## Citation

```bibtex
@misc{jiang2024llmengines,
  title = {LLM-Engines: A unified inference engine for large language models (LLMs) including open-source models (VLLM, SGLang, Together) and commercial models (OpenAI, Mistral, Claude)},
  author = {Dongfu Jiang},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/jdf-progLLM-Engines}},
}
```