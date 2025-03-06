# LLM-Engines

[Author: Dongfu Jiang](https://jdf-prog.github.io/), [Twitter](https://x.com/DongfuJiang/status/1833730295696334925), [PyPI Package](https://pypi.org/project/llm-engines/)

A unified inference engine for large language models (LLMs) including open-source models (VLLM, SGLang, Together) and commercial models (OpenAI, Mistral, Claude).

The correctness of the inference has been verified by comparing the outputs of the models with different engines when `temperature=0.0` and `max_tokens=None`.
For example, the outputs of a single model using 3 enginer (VLLM, SGLang, Together) will be the same when `temperature=0.0` and `max_tokens=None`.
Try examples below to see the outputs of different engines.

## News
- 2025-03-03: support `sleep` for vllm models, see [Sleep Mode](#sleep-mode) for more details.
- 2025-02-23: Support for vision input for all engines. See [Vision Input](#vision-input) for more details.
- 2025-02-19: Add support for `fireworks` api services, which provide calling for deepseek-r1 models with high speed.
- 2025-02-18: Add support for `grok` models.

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
- use vllm or sglang 
```python
from llm_engines import LLMEngine
model_name="Qwen/Qwen2.5-0.5B-Instruct"
llm = LLMEngine()
llm.load_model(
    model_name=model_name,
    num_workers=1, # number of workers
    num_gpu_per_worker=1, # tensor parallelism size for each worker
    engine="vllm", # or "sglang"
    use_cache=False
)
response = llm.call_model(model_name, "What is the capital of France?", temperature=0.0, max_tokens=None)
print(response)
```

- use together
```python
# export TOGETHER_API_KEY="your_together_api_key"
from llm_engines import LLMEngine
model_name="meta-llama/Llama-3-8b-chat-hf"
llm = LLMEngine()
llm.load_model(
    model_name=model_name, 
    engine="together", # or "openai", "mistral", "claude"
    use_cache=False
)
response = llm.call_model(model_name, "What is the capital of France?", temperature=0.0, max_tokens=None)
print(response)
```

- openai models
```python
# export OPENAI_API_KEY="your_openai_api_key"
from llm_engines import LLMEngine
model_name="gpt-3.5-turbo"
llm = LLMEngine()
llm.load_model(
    model_name=model_name, 
    engine="openai", # or "vllm", "together", "mistral", "claude"
    use_cache=False
)
response = llm.call_model(model_name, "What is the capital of France?", temperature=0.0, max_tokens=None)
print(response)
```

- grok models
```python
# export XAI_API_KEY="your_xai_api_key"
from llm_engines import LLMEngine
model_name="grok-2-latest"
llm = LLMEngine()
llm.load_model(
    model_name=model_name,
    engine="grok", # or "vllm", "together", "mistral", "claude"
    use_cache=False
)
response = llm.call_model(model_name, "What is the capital of France?", temperature=0.0, max_tokens=None)
print(response)
```

- mistral models
```python
# export MISTRAL_API_KEY="your_mistral_api_key"
from llm_engines import LLMEngine
model_name="mistral-large-latest"
llm = LLMEngine()
llm.load_model(
    model_name=model_name,
    engine="mistral", # or "vllm", "together", "openai", "claude"
    use_cache=False
)
response = llm.call_model(model_name, "What is the capital of France?", temperature=0.0, max_tokens=None)
print(response)
```

- claude models
```python
# export ANTHROPIC_API_KEY="your_claude_api_key"
from llm_engines import LLMEngine
model_name="claude-3-opus-20240229"
llm = LLMEngine()
llm.load_model(
    model_name=model_name,
    engine="claude", # or "vllm", "together", "openai", "mistral"
    use_cache=False
)
response = llm.call_model(model_name, "What is the capital of France?", temperature=0.0, max_tokens=None)
print(response)
```

- gemini models
```python
# export GEMINI_API_KEY="your_gemini_api_key"
from llm_engines import LLMEngine
model_name="gemini-1.5-flash"
llm = LLMEngine()
llm.load_model(
    model_name=model_name,
    engine="gemini", # or "vllm", "together", "openai", "mistral", "claude"
    use_cache=False
)
response = llm.call_model(model_name, "What is the capital of France?", temperature=0.0, max_tokens=None)
print(response)
```

- fireworks api
```python
```python
# export FIREWORKS_API_KEY="your_fireworks_api_key"
from llm_engines import LLMEngine
model_name="accounts/fireworks/models/deepseek-r1"
llm = LLMEngine()
llm.load_model(
    model_name=model_name,
    engine="fireworks", # or "vllm", "together", "openai", "mistral", "claude"
    use_cache=False
)
response = llm.call_model(model_name, "What is the capital of France?", temperature=0.0, max_tokens=None)
print(response)
```

### unload model
Remember to unload the model after using it to free up the resources. By default, all the workers will be unloaded after the program exits. If you want to use different models in the same program, you can unload the model before loading a new model, if that model needs gpu resources.
```python
llm.unload_model(model_name) # unload all the workers named model_name
llm.unload_model() # unload all the workers
```

### Multi-turn conversation
```python
from llm_engines import LLMEngine
model_name="Qwen/Qwen2.5-0.5B-Instruct"
llm = LLMEngine()
llm.load_model(
    model_name="Qwen/Qwen2.5-0.5B-Instruct", 
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
response = llm.call_model(model_name, messages, temperature=0.0, max_tokens=None)
print(response)
```
the messages should be in the format of 
- `[user_message, model_response, user_message, model_response, ...]`
- or in the format of openai's multi-turn conversation format.

### Vision Input
```python
from llm_engines import LLMEngine
from PIL import Image
import requests
from io import BytesIO
response = requests.get("https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg")
image = Image.open(BytesIO(response.content)).resize((256, 256))
image.save("./test.jpg")
messages_with_image = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What's in the image?"
            },
            {
                "type": "image",
                "image": image
            }
        ]
    }
] # the 'image' type is not offical format of openai API, LLM-Engines will convert it into image_url type internally
messages_with_image_url = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What's in the image?"
            },
            {
                "type": "image_url",
                "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"}
            }
        ]
    }
] # the 'image_url' type is the offical format of openai API
additional_args=[]
# engine="openai"; model_name="gpt-4o-mini"
# engine="claude"; model_name="claude-3-5-sonnet-20241022"
# engine="gemini"; model_name="gemini-2.0-flash"
# engine="grok"; model_name="grok-2-vision-latest"
# engine="sglang"; model_name="meta-llama/Llama-3.2-11B-Vision-Instruct"; additional_args=["--chat-template=llama_3_vision"] # refer to 
engine="vllm"; model_name="microsoft/Phi-3.5-vision-instruct"; additional_args=["--limit-mm-per-prompt", "image=2", "--max-model-len", "4096"] # refer to vllm serve api
llm = LLMEngine()
llm.load_model(
    model_name=model_name, 
    engine=engine, # or "vllm", "together", "mist
    use_cache=False,
    additional_args=additional_args,
)
response = llm.call_model(model_name, messages_with_image, temperature=0.0, max_tokens=None)
print(response)
response = llm.call_model(model_name, messages_with_image_url, temperature=0.0, max_tokens=None)
print(response)
```

### Sleep Mode
We support vllm's sleep mode if you want to save the GPU resources when the model is not used. (should have `vllm>=0.7.3`)
```python
import time
from llm_engines import LLMEngine
model_name="Qwen/Qwen2.5-0.5B-Instruct"
llm = LLMEngine()
llm.load_model(
    model_name=model_name,
    num_workers=1, # number of workers
    num_gpu_per_worker=1, # tensor parallelism size for each worker
    engine="vllm", # or "sglang"
    use_cache=False,
    additional_args=["--enable-sleep-mode"] # enable sleep mode
)
response = llm.call_model(model_name, "What is the capital of France?", temperature=0.0, max_tokens=None)
print(response)
llm.sleep_model(model_name) # sleep all the instances that named model_name
time.sleep(20) # check your GPU usage, it should be almost 0
llm.wake_up_model(model_name) # wake up all the instances that named model_name
response = llm.call_model(model_name, "What is the capital of France?", temperature=0.0, max_tokens=None)
```

### Batch inference
```python
from llm_engines import LLMEngine
model_name="Qwen/Qwen2.5-0.5B-Instruct"
llm = LLMEngine()
llm.load_model(
    model_name="Qwen/Qwen2.5-0.5B-Instruct", 
    num_workers=1, # number of workers
    num_gpu_per_worker=1, # tensor parallelism size for each worker
    engine="vllm", # or "sglang"
    use_cache=False
)
batch_messages = [
    "Hello", # user message 
    "Hello! It's nice to meet you. Is there something I can help you with, or would you like to chat?", # previous model response
    "What is the capital of France?" # user message
] * 100
response = llm.batch_call_model(model_name, batch_messages, num_proc=32, temperature=0.0, max_tokens=None)
print(response)
# List of responses [response1, response2, ...]
```
Example inference file: [`./examples/batch_inference_wildchat.py`](./examples/batch_inference_wildchat.py)
```bash
python examples/batch_inference_wildchat.py
```

**OpenAI Batch API**
by using the above code, it will automatically use the batch API for openai models. if you don't want to use the batch API and still want to use the normal API, set `disable_batch_api=True` when loading the model. `num_proc` will be ignored when using the batch API.

By using openai's batch API, you can get half the price of the normal API. The batch API is only available for the models with `max_batch_size > 1`.

LLM-Engines will calculates the hash of the inputs and generation parameters, and will only send new batch requests if the inputs and generation parameters are different from the previous requests. You can check a list of requested batch information in the [`~/llm_engines/generation_cache/openai_batch_cache/batch_submission_status.json`](~/llm_engines/generation_cache/openai_batch_cache/batch_submission_status.json) file.

### Parallel infernece throught huggingface dataset map
Check out [`./examples/mp_inference_wildchat.py`](./examples/mp_inference_wildchat.py) for parallel inference with multiple models.
```bash
python examples/mp_inference_wildchat.py
```

### Cache

if `use_cache=True`, all the queries and responses are cached in the `generation_cache` folder, no duplicate queries will be sent to the model.
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

### Worker initialization parameters (`load_model`)
- `model_name`: the model name, e.g., "Qwen/Qwen2.5-0.5B-Instruct" (required)
- `worker_addrs`: the list of worker addresses to use, if not provided, a new worker will be launched. If provided, it will use the existing workers (default: None)
- `num_workers`: the number of workers to use for the model (default: 1)
- `num_gpu_per_worker`: the number of GPUs to use for each worker (default: None)
- `engine`: the engine to use, one of {vllm, sglang, together, openai, mistral, claude, gemini} (default: "vllm")
- `additional_args`: list of str, additional arguments for launching the (vllm, sglang) worker, e.g., `["--max-model-len", "65536"]` (default: [])
- `use_cache`: whether to use the cache for the queries and responses (default: True)
- `cache_dir`: the cache directory, env variable `LLM_ENGINES_CACHE_DIR` (default: `~/llm-engines/generation_cache`)
- `overwrite_cache`: whether to overwrite the cache (default: False)
- `dtype`: the data type to use (default: "auto"; {auto,half,float16,bfloat16,float,float32})
- `quantization`: specify the quantization type, one of {aqlm,awq,deepspeedfp,tpu_int8,fp8,fbgemm_fp8,marlin,gguf,gptq_marlin_24,gptq_marlin,awq_marlin,gptq,squeezellm,compressed-tensors,bitsandbytes,qqq,experts_int8} (default: None)
- `max_retry`: the maximum number of retries for the request (default: None)
- `completion`: whether to use the completion API; If you use completion, (default: False)


### Generation parameters (`call_model`, `batch_call_model`)
- `inputs`: the list of inputs for the model; Either a list of strings or a list of dictionaries for multi-turn conversation in openai conversation format; If `completion` is True, it should be a single string (required)
- `top_p`: the nucleus sampling parameter, 0.0 means no sampling (default: 1.0)
- `temperature`: the randomness of the generation, 0.0 means deterministic generation (default: 0.0)
- `max_tokens`: the maximum number of tokens to generate, `None` means no limit (default: None)
- `timeout`: the maximum time to wait for the response, `None` means no limit (default: 300)
- `frequency_penalty`: Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim. (default: 0.0)
- `presence_penalty`: Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics. (default: 0.0)
- `n`: Number of completions to generate for each prompt. (**only vllm, sglang, openai have this feature**) (default: 1)
- `stream`: Whether to stream the response or not. If True, `n` will be ignored. (default: False)
- `conv_system_msg`: The system message for multi-turn conversation; If the meessage contains a system message, this parameter will be overwritten (default: None)
- `logprobs`: Whether to return the log probabilities of the generated tokens, True/False/None (default: None)
- all the other parameters that are supported by different engines.
    - for openai and sglang, check out [openai](https://platform.openai.com/docs/api-reference/chat)
    - for extra paramters of vllm, check out [vllm](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters)

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
    model_name="Qwen/Qwen2.5-0.5B-Instruct", 
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
- Bug: [issue](https://github.com/vllm-project/vllm/issues/7196) of `vllm==0.5.4` when num_workers > 1, use `vllm==0.5.5` instead.
- Try not load the same openai models with different cache directories, the current code only loads the cache from the first provided cache directory. But when writing the cache, it will write to different cache directories correspondingly. This might cause some confusion when using.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=jdf-prog/LLM-Engines&type=Date)](https://star-history.com/#jdf-prog/LLM-Engines&Date)

## Citation
```bibtex
@misc{jiang2024llmengines,
  title = {LLM-Engines: A unified and parallel inference engine for large language models},
  author = {Dongfu Jiang},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/jdf-progLLM-Engines}},
}
```
