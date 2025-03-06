import os
import time
import torch
import random
import json
import openai
import vllm
import signal
import regex as re
from pathlib import Path
from typing import List
from .utils import SubprocessMonitor, ChatTokenizer, with_timeout, get_function_arg_names
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
worker_initiated = False
vllm_version = vllm.__version__

chat_tokenizers = {}
def launch_vllm_worker(
    model_name: str,
    num_gpus: int=None,
    gpu_ids: List[int]=None,
    dtype: str="auto",
    quantization: str=None,
    port: int=34200,
    host: str="127.0.0.1",
    root_path: str=None,
    additional_args: List[str]=[],
) -> str:
    """
    Launch a model worker and return the address
    Args:
        model_name: the model name to launch
    Returns:
        the address of the launched model
    """
    print(f"Launching model {model_name}")
    worker_addr = f"http://{host}:{port}"
    log_file = Path(os.path.abspath(__file__)).parent / "logs" / f"{model_name}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if gpu_ids:
        num_gpus = len(gpu_ids)
    else:
        if not num_gpus:
            num_gpus = torch.cuda.device_count()
            print(f"Warning: num_gpus or gpu_ids not provided, using {num_gpus} GPUs")
        gpu_ids = list(range(num_gpus))
    env = os.environ.copy()
    # Set the CUDA_VISIBLE_DEVICES environment variable
    env["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu_id) for gpu_id in gpu_ids])
    if "gemma-2" in model_name:
        env["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
    else:
        env["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    env["VLLM_SERVER_DEV_MODE"] = "1"
    # print(num_gpus, gpu_ids)
    
    model_path = Path(model_name)
    if model_path.exists() and ((model_path / "config.json").exists() or (model_path / "adapter_config.json").exists()):
        if (model_path / "adapter_config.json").exists():
            print(f"Loading local model {model_name} with adapter")
            use_lora = True
            with open(model_path / "adapter_config.json") as f:
                adapter_config = json.load(f)
            adapter_path = model_path
            base_model_name_or_path = adapter_config["base_model_name_or_path"]
        elif (model_path / "config.json").exists():
            print(f"Loading local model {model_name}")
            use_lora = False
            adapter_path = None
            base_model_name_or_path = model_name
        else:
            raise ValueError(f"no config.json or adapter_config.json found in model {model_name}")
    else:
        # check whether there is a adapter_config.json
        api = HfApi()
        model_info = api.model_info(model_name)
        model_files = [x.rfilename for x in model_info.siblings]
        if "adapter_config.json" in model_files:
            use_lora = True
            adapter_path = snapshot_download(model_name)
            adapter_config_path = Path(adapter_path) / "adapter_config.json"
            with open(adapter_config_path) as f:
                adapter_config = json.load(f)
            base_model_name_or_path = adapter_config["base_model_name_or_path"]
            print(f"Loading model from Hugging Face {model_name} with adapter")
        elif "config.json" in model_files:
            use_lora = False
            adapter_path = None
            base_model_name_or_path = model_name
            print(f"Loading model from Hugging Face {model_name}")
        else:
            raise ValueError(f"no config.json or adapter_config.json found in model {model_name}")
        
    # python -m vllm.entrypoints.openai.api_server \
    # --model meta-llama/Llama-2-7b-hf \
    # --enable-lora \
    # --lora-modules sql-lora=~/.cache/huggingface/hub/models--yard1--llama-2-7b-sql-lora-test/
    if use_lora:
        lora_args = [
            "--enable-lora",
            "--lora-modules", f"{model_name}={adapter_path}",
            "--max-loras", "1",
            "--max-lora-rank", str(adapter_config["r"])
        ]
    else:
        lora_args = []
    if quantization:
        available_quantizations = "aqlm,awq,deepspeedfp,tpu_int8,fp8,fbgemm_fp8,marlin,gguf,gptq_marlin_24,gptq_marlin,awq_marlin,gptq,squeezellm,compressed-tensors,bitsandbytes,qqq,experts_int8"
        available_quantizations = available_quantizations.split(",")
        if quantization not in available_quantizations:
            raise ValueError(f"quantization {quantization} not in available quantizations: {available_quantizations}")
        lora_args += ["--quantization", quantization]
        if quantization == "bitsandbytes":
            lora_args += ["--load-format", "bitsandbytes", "--enforce-eager"]
    # python -m vllm.entrypoints.openai.api_server --model NousResearch/Meta-Llama-3-8B-Instruct --dtype auto --api-key token-abc123
    proc = SubprocessMonitor([
        "vllm", "serve",
        base_model_name_or_path,
        "--dtype", dtype,
        "--api-key", "vllm-engine-token",
        "--port", str(port),
        "--host", host,
        "--tensor-parallel-size", str(num_gpus),
        "--disable-log-requests",
        "--trust-remote-code",
        # "--enable-sleep-mode"
    ] + (["--root-path", root_path] if root_path else [])
    + lora_args + additional_args, env=env)
    print(f"Launched VLLM model {model_name} at address {worker_addr} with CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
    if model_name not in chat_tokenizers:
        chat_tokenizers[model_name] = ChatTokenizer(base_model_name_or_path)
    if base_model_name_or_path not in chat_tokenizers:
        chat_tokenizers[base_model_name_or_path] = ChatTokenizer(base_model_name_or_path)
    return f"http://127.0.0.1:{port}", proc

def call_vllm_worker(messages, model_name, worker_addrs, timeout:int=300, conv_system_msg=None, **generate_kwargs) -> str:
    global worker_initiated
    global chat_tokenizers
    if "max_new_tokens" in generate_kwargs:
        if "max_tokens" not in generate_kwargs:
            generate_kwargs["max_tokens"] = generate_kwargs["max_new_tokens"]
        del generate_kwargs["max_new_tokens"]
    # try:
    #     if model_name not in chat_tokenizers:
    #         chat_tokenizers[model_name] = ChatTokenizer(model_name)
    #     chat_tokenizer = chat_tokenizers[model_name]
    #     prompt = chat_tokenizer(chat_messages)
    # except Exception as e:
    #     pass
    
    # change messages to openai format
    if conv_system_msg:
        chat_messages = [{"role": "system", "content": conv_system_msg}] + messages
    else:
        chat_messages = messages
        
    worker_addr = random.choice(worker_addrs)
    
    client = openai.OpenAI(
        base_url=f"{worker_addr}/v1",
        api_key="vllm-engine-token",
    )

    args_names, kwargs_names = get_function_arg_names(client.chat.completions.create)
    extra_body_params = {}
    for key in list(generate_kwargs.keys()):
        if key not in args_names + kwargs_names:
            extra_body_params[key] = generate_kwargs[key]
            del generate_kwargs[key]
    generate_kwargs["extra_body"] = extra_body_params
    
    stream = generate_kwargs.get("stream", False)
    if stream:
        generate_kwargs.pop("n", None)
    
    # print(generate_kwargs)
    @with_timeout(timeout)
    def get_response():
        while True:
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=chat_messages,
                    **generate_kwargs,
                )
                break
            except openai.APIConnectionError as e:
                if not worker_initiated:
                    time.sleep(5)
                    continue
                print(f"API connection error: {e}")
                time.sleep(5)
                continue
            except openai.BadRequestError as e:
                error_response = e.response.json()
                if error_response['code'] == 400:
                    pattern = r"This model's maximum context length is (\d+) tokens. However, you requested (\d+) tokens \((\d+) in the messages, (\d+) in the completion\). Please reduce the length of the messages or completion."
                    res = re.match(pattern, error_response['message'])
                    if res:
                        max_context_length = int(res.group(1))
                        num_tokens_requested = int(res.group(2))
                        num_tokens_in_messages = int(res.group(3))
                        num_tokens_in_completion = int(res.group(4))
                        
                        new_max_tokens = num_tokens_in_completion - (num_tokens_requested - max_context_length)
                        if new_max_tokens <= 0:
                            raise e
                        print(f"Reducing max_tokens to {new_max_tokens}, and retrying")
                        generate_kwargs["max_tokens"] = new_max_tokens
                        continue
                    else:
                        raise e

        if not stream:
            if "logprobs" not in generate_kwargs or not generate_kwargs["logprobs"]:
                if len(completion.choices) > 1:
                    return [c.message.content for c in completion.choices]
                else:
                    return completion.choices[0].message.content
            else:
                if len(completion.choices) > 1:
                    return [c.message.content for c in completion.choices], [c.logprobs.dict() for c in completion.choices]
                else:
                    return completion.choices[0].message.content, completion.choices[0].logprobs.dict()
        else:
            def generate_stream():
                for chunk in completion:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content
            return generate_stream()
    
    return get_response()
    
def call_vllm_worker_completion(prompt:str, model_name, worker_addrs, timeout:int=300, **generate_kwargs) -> str:
    global worker_initiated
    
    worker_addr = random.choice(worker_addrs)
    
    client = openai.OpenAI(
        base_url=f"{worker_addr}/v1",
        api_key="vllm-engine-token",
    )
    
    stream = generate_kwargs.get("stream", False)
    if stream:
        generate_kwargs.pop("n", None)
        
    args_names, kwargs_names = get_function_arg_names(client.completions.create)
    extra_body_params = {}
    for key in list(generate_kwargs.keys()):
        if key not in args_names + kwargs_names:
            extra_body_params[key] = generate_kwargs[key]
            del generate_kwargs[key]
    # generate_kwargs["extra_body"] = extra_body_params
    
    @with_timeout(timeout)
    def get_response():
        while True:
            try:
                completion = client.completions.create(
                    model=model_name,
                    prompt=prompt,
                    **generate_kwargs,
                )
                break
            except openai.APIConnectionError as e:
                if not worker_initiated:
                    time.sleep(5)
                    continue
                print(f"API connection error: {e}")
                time.sleep(5)
                continue
        if not stream:
            if len(completion.choices) > 1:
                return [c.text for c in completion.choices]
            else:
                return completion.choices[0].text
        else:
            def generate_stream():
                for chunk in completion:
                    if chunk.choices[0].text is not None:
                        yield chunk.choices[0].text
            return generate_stream()
    return get_response()