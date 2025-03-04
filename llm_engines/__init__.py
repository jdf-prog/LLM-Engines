import os
import sys
import json
import random
import atexit
import psutil
import hashlib
import requests
import subprocess
from packaging import version
from functools import partial
from .utils import retry_on_failure, convert_messages_wrapper, SubprocessMonitor, MaxRetriesExceededError, max_retry_wrapper  
from .cache import get_batch_cache_dir, generation_cache_wrapper
from typing import Union, List
from tqdm import tqdm

import importlib.util
flash_attn = importlib.util.find_spec("flash_attn")
if not flash_attn:
    print("Warning: flash_attn not found, recommend to install flash_attn for better performance")
    print("Simple Command: pip install flash_attn --no-build-isolation")
    print("Please refer to https://github.com/Dao-AILab/flash-attention for detailed installation instructions")

ENGINES = ["vllm", "sglang", "openai", "gemini", "mistral", "together", "claude"]
all_workers = []
verbose = False
def set_verbose(value):
    global verbose
    verbose = value
    
class WorkerInstance:
    def __init__(self, model_name, worker_addr, proc, gpu_ids=None):
        self.model_name = model_name
        self.worker_addr = worker_addr
        self.proc = proc
        self.gpu_ids = gpu_ids
    
    def __str__(self):
        return f"WorkerInstance(model_name={self.model_name}, worker_addr={self.worker_addr}, proc={self.proc}, gpu_ids={self.gpu_ids})"
    
    def __repr__(self):
        return self.__str__()


class ModelWorker:
    def __init__(
        self,
        model_name, 
        worker_addrs=None, 
        cache_dir=None, 
        use_cache=True,
        overwrite_cache=False,
        completion=False, 
        num_workers=1,
        num_gpu_per_worker=None,
        gpu_ids=None,
        dtype="auto",
        quantization=None,
        engine="vllm",
        additional_args=[],
        max_retry=None,
        verbose=False,
    ):
        """
        Return a function that calls the model worker, takes a list of messages (user, gpt, user, ...) and returns the generated text
        Args:
            model_name: model name
            worker_addrs: worker addresses, if None, launch local workers
            cache_dir: cache directory
            use_cache: use cache or not. Cache is on the hash of the input message.
            overwrite_cache: overwrite cache or not. If True, previous cache will be overwritten.
            completion: use completion or not (use chat by default)
            num_workers: number of workers
            num_gpu_per_worker: number of gpus per worker
            dtype: data type
            engine: engine name
            additional_args: additional arguments for launching the worker (vllm, sglang)
        """
        self.model_name = model_name
        self.worker_addrs = worker_addrs
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.overwrite_cache = overwrite_cache
        self.completion = completion
        self.num_workers = num_workers
        self.num_gpu_per_worker = num_gpu_per_worker
        self.gpu_ids = gpu_ids
        self.dtype = dtype
        self.quantization = quantization
        self.engine = engine
        self.additional_args = additional_args
        self.max_retry = max_retry
        self.verbose = verbose
        self.worker_instances = [] # cuda workers instances
        self.is_sleeping = False
        
        if engine == "openai":
            from .openai_text import call_worker_openai, call_worker_openai_completion
            call_model_worker = call_worker_openai if not completion else call_worker_openai_completion
        elif engine == "gemini":
            if completion:
                raise ValueError(f"Engine {engine} does not support completion")
            from .gemini import call_worker_gemini
            call_model_worker = call_worker_gemini
        elif engine == "claude":
            if completion:
                raise ValueError(f"Engine {engine} does not support completion")
            from .claude import call_worker_claude
            call_model_worker = call_worker_claude
        elif engine == "mistral":
            if completion:
                raise ValueError(f"Engine {engine} does not support completion")
            from .mistral import call_worker_mistral
            call_model_worker = call_worker_mistral
        elif engine == "together":
            from .together import call_worker_together, call_worker_together_completion
            call_model_worker = call_worker_together if not completion else call_worker_together_completion
        elif engine == "grok":
            from .grok import call_worker_grok, call_worker_grok_completion
            call_model_worker = call_worker_grok if not completion else call_worker_grok_completion 
        elif engine == "fireworks":
            from .fireworks import call_worker_fireworks, call_worker_fireworks_completion
            call_model_worker = call_worker_fireworks if not completion else call_worker_fireworks_completion
        elif engine in ["vllm", "sglang"]:
            assert num_gpu_per_worker is not None, "num_gpu_per_worker must be provided for vllm and sglang"
            if engine == "vllm":
                from .vllm import launch_vllm_worker, call_vllm_worker, call_vllm_worker_completion
                call_worker_func = call_vllm_worker if not completion else call_vllm_worker_completion
                launch_worker_func = launch_vllm_worker
            elif engine == "sglang":
                from .sglang import launch_sglang_worker, call_sglang_worker, call_sglang_worker_completion
                call_worker_func = call_sglang_worker if not completion else call_sglang_worker_completion
                launch_worker_func = launch_sglang_worker
            else:
                raise ValueError(f"Internal error: engine {engine} not supported")
            if worker_addrs is None:
                import torch
                
                print(f"Launching model worker {model_name} locally")
                worker_addrs = []
                total_gpus = torch.cuda.device_count()
                if gpu_ids:
                    gpu_ids = [int(gpu_id) for gpu_id in gpu_ids]
                    assert len(gpu_ids) <= total_gpus, f"Error: number of gpus {len(gpu_ids)} is greater than total gpus {total_gpus}"
                    total_gpus = len(gpu_ids)
                if total_gpus < num_workers * num_gpu_per_worker:
                    if total_gpus >= num_gpu_per_worker:
                        print(f"Warning: total gpus ({total_gpus}) is less than num_workers * num_gpu_per_worker ({num_workers * num_gpu_per_worker}), using {total_gpus // num_gpu_per_worker} workers instead")
                        num_workers = total_gpus // num_gpu_per_worker
                    else:
                        print(f"Error: total gpus ({total_gpus}) is less than num_gpu_per_worker ({num_gpu_per_worker}), exiting...")
                        sys.exit(1)
                if not gpu_ids:
                    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
                        gpus_ids = os.environ.get("CUDA_VISIBLE_DEVICES").split(",")
                        gpu_ids = [int(gpu_id) for gpu_id in gpus_ids]
                    else:
                        gpu_ids = list(range(total_gpus))
                start_port = random.randint(31000, 32000)
                for i in range(num_workers):
                    worker_addr, proc = launch_worker_func(model_name, 
                        num_gpus=num_gpu_per_worker, 
                        gpu_ids=gpu_ids[i*num_gpu_per_worker:(i+1)*num_gpu_per_worker], 
                        port=start_port+i*10,
                        dtype=dtype, quantization=quantization, additional_args=additional_args)
                    worker = WorkerInstance(model_name, worker_addr, proc, gpu_ids=gpu_ids[i*num_gpu_per_worker:(i+1)*num_gpu_per_worker])
                    worker_addrs.append(worker_addr)
                    all_workers.append(worker)
                    self.worker_instances.append(worker)
            else:
                if verbose:
                    print(f"Using existing worker at {worker_addrs}")
                if not isinstance(worker_addrs, list):
                    worker_addrs = [worker_addr]
            call_model_worker = partial(call_worker_func, worker_addrs=worker_addrs)   
            self.worker_addrs = worker_addrs
            self.gpu_ids = gpu_ids
        else:
            raise ValueError(f"Engine {engine} not supported, available engines: {ENGINES}")
        
        # wrap the call_model_worker with the model_name and other arguments
        call_model_worker = partial(call_model_worker, model_name=model_name)
        # test local worker connection
        if not completion:
            test_response = call_model_worker([{"role": "user", "content": "Hello"}], temperature=0, max_tokens=256, timeout=None)
        else:
            test_response = call_model_worker("Hello", temperature=0, max_tokens=256, timeout=None)
        if not test_response:
            print("Error: failed to connect to the worker, exiting...")
            for worker in self.worker_instances:
                cleanup_process(worker)
            sys.exit(1)
        else:
            if verbose:
                print(f"Successfully connected to the workers")
                print("Test prompt: \n", "Hello")
                print("Test response: \n", test_response)
            
        # add cache wrapper
        if use_cache:
            call_model_worker = generation_cache_wrapper(call_model_worker, model_name, cache_dir, overwrite_cache)
        else:
            if verbose:
                print("Cache is disabled")
        call_model_worker = retry_on_failure(call_model_worker, num_retries=max_retry)
        call_model_worker = convert_messages_wrapper(call_model_worker, is_completion=completion)
        set_do_cleanup(True)
        set_verbose(verbose)
        self.call_model_worker = call_model_worker
        
    def __call__(
        self,
        messages:Union[str, List[str], List[dict]],
        *args, **kwds
    ):
        if self.is_sleeping:
            print("Warning: Worker is sleeping, waking up...")
            self.wake_up_worker()
        return self.call_model_worker(messages, *args, **kwds)
    
    def sleep_worker(self, level=1):
        if self.engine in ["vllm"]:
            # vllm version should be >= 0.7.3
            from .vllm import vllm_version
            if version.parse(vllm_version) < version.parse("0.7.3"):
                raise ValueError(f"vllm version {vllm_version} does not support sleep mode, please upgrade to >= 0.7.3")
            for worker in self.worker_instances:
                response = requests.post(worker.worker_addr + "/sleep",
                                    data={"level": level})
                print(worker.worker_addr + "/sleep")
                print(response)
                assert response.status_code == 200
                print(f"Worker {worker} is sleeping")
            self.is_sleeping = True
        elif self.engine in ["sglang"]:
            raise NotImplementedError(f"Engine {self.engine} does not support sleep")
            
    def wake_up_worker(self):
        if not self.is_sleeping:
            return
        if self.engine in ["vllm"]:
            # vllm version should be >= 0.7.3
            from .vllm import vllm_version
            if version.parse(vllm_version) < version.parse("0.7.3"):
                raise ValueError(f"vllm version {vllm_version} does not support sleep mode, please upgrade to >= 0.7.3")
            for worker in self.worker_instances:
                response = requests.post(worker.worker_addr + "/wake_up")
                assert response.status_code == 200
                print(f"Worker {worker} is woken up")
        elif self.engine in ["sglang"]:
            raise NotImplementedError(f"Engine {self.engine} does not support wake up")
    
    def __str__(self):
        return f"ModelWorker(model_name={self.model_name}, num_workers={self.num_workers}, num_gpu_per_worker={self.num_gpu_per_worker}, worker_addrs={self.worker_addrs}, cache_dir={self.cache_dir}, use_cache={self.use_cache}, overwrite_cache={self.overwrite_cache}, completion={self.completion}, dtype={self.dtype}, quantization={self.quantization}, engine={self.engine}, additional_args={self.additional_args}, max_retry={self.max_retry}, verbose={self.verbose})"
    
    def __repr__(self):
        return self.__str__()


do_cleanup = True
def set_do_cleanup(value):
    global do_cleanup
    do_cleanup = value
    
def kill_process_and_children(pid):
    # check if the process is still alive
    if 'psutil' not in sys.modules:
        # possibly the main process is in the final stage of termination, no need to kill the child processes
        return None
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        print(f"No process with PID {pid} found.")
        return

    children = parent.children(recursive=True)
    for child in children:
        try:
            child.kill()
            # print(f"Killed child process {child.pid}")
        except psutil.NoSuchProcess:
            print(f"Child process {child.pid} already terminated.")
    
    try:
        parent.kill()
        # print(f"Killed parent process {pid}")
    except psutil.NoSuchProcess:
        print(f"Parent process {pid} already terminated.")
    return True
        
def cleanup_process(worker:Union[ModelWorker, WorkerInstance, SubprocessMonitor, subprocess.Popen]):
    if isinstance(worker, ModelWorker):
        for worker_instance in worker.worker_instances:
            cleanup_process(worker_instance)
        return
    if isinstance(worker, WorkerInstance):
        proc = worker.proc.proc
    elif isinstance(worker, SubprocessMonitor):
        proc = worker.proc
    elif isinstance(worker, subprocess.Popen):
        proc = worker
    else:
        raise ValueError(f"Unknown process type {type(proc)}")
    killed = kill_process_and_children(proc.pid)
    if verbose and killed:
        print(f"Model Worker terminated: {worker} ")
    return killed

@atexit.register
def cleanup_all_workers():
    if not do_cleanup:
        return
    for worker in all_workers:
        cleanup_process(worker)
    if all_workers and verbose:
        print("All workers terminated.")
    all_workers.clear()


class LLMEngine:
    
    def __init__(self, verbose=False, num_gpus: int = None, gpu_ids: Union[List[int], str] = None):
        self.workers = []
        self.loaded_model_worker = {}
        import torch
        self.verbose = verbose
        total_gpus = torch.cuda.device_count()
        if gpu_ids:
            assert isinstance(gpu_ids, (list, str)), "passed gpu_ids must be a list or a string"
            if isinstance(gpu_ids, str):
                gpu_ids = [int(gpu_id) for gpu_id in gpu_ids.split(",")]
            assert all(isinstance(gpu_id, int) for gpu_id in gpu_ids), "passed gpu_ids must be a list of integers"
            assert len(gpu_ids) <= total_gpus, f"Error: passed gpu_ids {gpu_ids} is greater than total gpus {total_gpus}"
        else:
            if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
                gpu_ids = [int(gpu_id) for gpu_id in os.environ.get("CUDA_VISIBLE_DEVICES").split(",")]
            else:
                gpu_ids = list(range(total_gpus))
            if num_gpus is not None:
                num_gpus = int(num_gpus)
                assert num_gpus <= total_gpus, f"Error: passed num_gpus {num_gpus} is greater than total gpus {total_gpus}"
                gpu_ids = gpu_ids[:num_gpus]
        self.gpu_ids = gpu_ids
        self.num_gpus = len(gpu_ids)
        if verbose:
            print(f"LLMEngine initialized with {self.num_gpus} GPUs: {gpu_ids}")
        
    def get_available_gpu_ids(self):
        worker_used_gpu_ids = []
        for worker in self.workers:
            for worker_instance in worker.worker_instances:
                worker_used_gpu_ids.extend(worker_instance.gpu_ids)
        available_gpu_ids = [gpu_id for gpu_id in self.gpu_ids if gpu_id not in worker_used_gpu_ids]
        return available_gpu_ids
    
    def load_model(
        self,
        model_name,
        worker_addrs=None,
        cache_dir=None,
        use_cache=True,
        overwrite_cache=False,
        completion=False,
        num_workers=1,
        num_gpu_per_worker=None,
        dtype="auto",
        quantization=None,
        engine="vllm",
        additional_args=[],
        max_retry=None,
        verbose=None
    ):
        """
        Load a model
        Args:
            model_name: model name
            worker_addrs: worker addresses, if None, launch local workers
            cache_dir: cache directory
            use_cache: use cache or not. Cache is on the hash of the input message.
            overwrite_cache: overwrite cache or not. If True, previous cache will be overwritten.
            completion: use completion or not (use chat by default)
            num_workers: number of workers
            num_gpu_per_worker: number of gpus per worker
            dtype: data type
            engine: engine name
            additional_args: additional arguments for launching the worker (vllm, sglang)
            max_retry: maximum number of retries
            verbose: verbose
        """
        verbose = self.verbose or verbose
        if self.workers:
            print("Warning: previous workers are not cleaned up, please call unload_model() to clean up previous workers")
        self.model_name = model_name
        available_gpu_ids = self.get_available_gpu_ids()
        if engine in ["vllm", "sglang"]:
            if not num_gpu_per_worker:
                print("Warning: num_gpu_per_worker not provided, using 1 GPU per worker")
                num_gpu_per_worker = 1
            num_required_gpus = num_workers * num_gpu_per_worker
            if len(available_gpu_ids) < num_required_gpus:
                print("Error: No available GPU to launch the model worker")
                print("Provided GPU IDs for this LLMEngine class: ", self.gpu_ids)
                print("Used GPU IDs for all workers: ", [worker_instance.gpu_ids for worker in self.workers for worker_instance in worker.worker_instances])
                print("Available GPU IDs: ", available_gpu_ids)
                print("Number of required GPUs: ", num_required_gpus)
                raise ValueError("Not enought available GPU to launch the model worker")
            gpu_ids = available_gpu_ids[:num_required_gpus]
        else:
            num_required_gpus = num_workers
            gpu_ids = None
            
        model_worker = ModelWorker(
            model_name, 
            worker_addrs=worker_addrs, 
            cache_dir=cache_dir, 
            use_cache=use_cache,
            overwrite_cache=overwrite_cache,
            completion=completion, 
            num_workers=num_workers,
            num_gpu_per_worker=num_gpu_per_worker,
            gpu_ids=gpu_ids,
            dtype=dtype,
            quantization=quantization,
            engine=engine,
            additional_args=additional_args,
            max_retry=max_retry,
            verbose=verbose,
        )
        self.workers.append(model_worker)
        self.loaded_model_worker[model_name] = model_worker
        return model_worker

    def sleep_model(
        self,
        model_name,
        level=1
    ):
        model_worker = self.loaded_model_worker.get(model_name)
        model_worker.sleep_worker(level=level)
        
    def wake_up_model(
        self,
        model_name,
    ):
        model_worker = self.loaded_model_worker.get(model_name)
        model_worker.wake_up_worker()
    
    def call_model(
        self, 
        model_name, 
        messages:Union[List[str], List[dict], str],
        timeout:int=60, 
        conv_system_msg=None, 
        **generate_kwargs
    ):
        """
        Call a model
        Args:
            model_name: model name
            messages: list of messages in openai format
            timeout: timeout
            conv_system_msg: conversation system message
            generate_kwargs: generation arguments
        """
        call_model_worker = self.loaded_model_worker.get(model_name)
        if call_model_worker is None:
            raise ValueError(f"Model {model_name} not loaded, please call load_model() first")
        try:
            return call_model_worker(messages, timeout=timeout, conv_system_msg=conv_system_msg, **generate_kwargs)
        except MaxRetriesExceededError as e:
            print(e)
            return None
    
    def batch_call_model(
        self,
        model_name,
        batch_messages:List[Union[List[str], List[dict], str]],
        timeout:int=60,
        conv_system_msg=None,
        num_proc=8,
        desc=None,
        disable_batch_api=False,
        max_batch_size=None,
        **generate_kwargs
    ):
        """
        Batch call a model
        Args:
            model_name: model name
            batch_messages: list of list of messages in openai format or list of strings
            timeout: timeout
            conv_system_msg: conversation system message
            num_proc: number of processes
            generate_kwargs: generation arguments
        """
        supported_batch_api_engines = ["openai", "claude"]
        model_worker = self.loaded_model_worker.get(model_name)
        engine = model_worker.engine
        if engine not in supported_batch_api_engines or disable_batch_api:
            if model_worker is None:
                raise ValueError(f"Model {model_name} not loaded, please call load_model() first")
            
            batch_cache_dir = get_batch_cache_dir(model_name, None)
            to_write_batch_messages = [
                {"input": message, "generation_kwargs": generate_kwargs} for message in batch_messages
            ]
            hash_str = hashlib.md5("".join([
                str(x) for x in to_write_batch_messages
            ]).encode()).hexdigest()
            hash_result_file = batch_cache_dir / f"{hash_str}_batch_results.jsonl"
            if hash_result_file.exists():
                results = []
                with open(hash_result_file, "r") as f:
                    for line in f:
                        message = json.loads(line)
                        results.append(message["output"])
            else:
                from functools import partial
                from multiprocessing import Pool
                num_proc = min(num_proc, len(batch_messages))
                if model_worker.is_sleeping:
                    print("Warning: Worker is sleeping, waking up...")
                    model_worker.wake_up_worker()
                call_model_worker_mp = partial(model_worker.call_model_worker, timeout=timeout, conv_system_msg=conv_system_msg, **generate_kwargs)
                call_model_worker_mp = partial(max_retry_wrapper, call_model_worker_mp)
                with Pool(num_proc) as p:
                    results = list(tqdm(p.imap(call_model_worker_mp, batch_messages), total=len(batch_messages), desc=desc or "LLMEngine Batch Inference"))
                if results:
                    for i, message in enumerate(to_write_batch_messages):
                        message["output"] = results[i]
                    with open(batch_cache_dir / f"{hash_str}_batch_results.jsonl", "w") as f:
                        for message in to_write_batch_messages:
                            f.write(json.dumps(message) + "\n")
        else:
            if engine == "openai":
                print("Using OpenAI batch API")
                from .openai_text import openai_batch_request, check_batch_status, get_batch_progress, get_batch_result
                batch_request_func = openai_batch_request
            elif engine == "claude":
                print("Using Claude batch API")
                from .claude import claude_batch_request, check_batch_status, get_batch_progress, get_batch_result
                batch_request_func = claude_batch_request
            else:
                raise ValueError(f"Engine {engine} not supported for batch API")
            if max_batch_size is None:
                results = batch_request_func(model_name, batch_messages, conv_system_msg=conv_system_msg, desc=desc, **generate_kwargs)
            else:
                # using multiprocess to submit batch request per batch
                from functools import partial
                import time
                
                max_slots = num_proc
                all_batch_inputs = [
                    batch_messages[i:i+max_batch_size] for i in range(0, len(batch_messages), max_batch_size)
                ]
                # submit detach jobs
                batch_ids = [None] * len(all_batch_inputs)
                # wait for all jobs to finish and periodically check the status
                idx = 0
                tqdm_bar = tqdm(total=len(batch_ids), desc=desc or "LLMEngine Batch Inference")
                all_batch_status = [check_batch_status(batch_id)['status'] if batch_id is not None else "pending" for batch_id in batch_ids]
                while True:
                    batch_id = batch_ids[idx]
                    if batch_id is None:
                        cur_slots = len([bstatus for bstatus in all_batch_status if bstatus not in ['pending', 'completed', 'cancelled', 'canceled', "failed", "expired"]])
                        if cur_slots < max_slots:
                            batch_id = batch_request_func(model_name, all_batch_inputs[idx], conv_system_msg=conv_system_msg, desc=desc, **generate_kwargs, detach=True)
                            batch_ids[idx] = batch_id
                        else:
                            tqdm_bar.n = 0
                            tqdm_bar.total = len(all_batch_inputs[idx])
                            tqdm_bar.desc = "pending" + f" (batch {idx+1}/{len(batch_ids)})"
                            tqmd_postfix = tqdm_postfix = {
                                "completed": 0,
                                "total": len(all_batch_inputs[idx]),
                                "failed": 0,
                            }
                            tqdm_bar.set_postfix_str(tqmd_postfix)
                            tqdm_bar.refresh()
                            idx = (idx + 1) % len(batch_ids)
                            time.sleep(5)
                            continue
                        
                    n, total, tqdm_postfix, cur_batch_status = get_batch_progress(batch_id)
                    tqdm_bar.set_postfix_str(tqdm_postfix)
                    tqdm_bar.n = n
                    tqdm_bar.total = total
                    tqdm_bar.desc = cur_batch_status + f" (batch {idx+1}/{len(batch_ids)})"
                    tqdm_bar.refresh()
                    all_batch_status[idx] = cur_batch_status
                    if all(status == "completed" for status in all_batch_status):
                        tqdm_bar.close()
                        break
                    idx = (idx + 1) % len(batch_ids)
                    time.sleep(10)
                
                # collect results
                all_batch_results = []
                for i, batch_id in enumerate(batch_ids):
                    batch_results = get_batch_result(batch_id, generate_kwargs=generate_kwargs)
                    if batch_results:
                        all_batch_results.extend(batch_results)
                    else:
                        raise ValueError(f"Warning: batch {batch_id} has no results")
                results = all_batch_results
                    
        return results
    
    def __call__(self, *args, **kwds):
        return self.call_model(*args, **kwds)
                    
    def unload_model(self, model_name=None):
        to_remove_local_workers = []
        to_remove_global_workers = []
        for worker in self.workers:
            if model_name is None or worker.model_name == model_name:
                print(f"Unloading model worker: {worker}")
                cleanup_process(worker) 
                if worker in all_workers:
                    to_remove_global_workers.append(worker)
                if worker in self.workers:
                    to_remove_local_workers.append(worker)
        for worker in to_remove_global_workers:
            all_workers.remove(worker)
        for worker in to_remove_local_workers:
            self.workers.remove(worker)
        
    def __del__(self):
        pass
        
        