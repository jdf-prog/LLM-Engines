import fire
import random
import os
from typing import List, Union

class Cli:
    def __init__(self):
        pass

    def serve(self,
        model_name: str,
        engine: str="vllm",
        num_gpus: int=1,
        gpu_ids: List[int]=None,
        dtype: str="auto",
        quantization: str=None,
        port: Union[int,None] =None,
        host: str="127.0.0.1",
    ) -> str:
        from . import set_do_cleanup
        set_do_cleanup(False)
        assert engine in ["vllm", "sglang"]
        if engine == "vllm":
            from .vllm import launch_vllm_worker
            launch_worker_func = launch_vllm_worker
        elif engine == "sglang":
            from .sglang import launch_sglang_worker
            launch_worker_func = launch_sglang_worker
        if port is None:
            print("Warning: port not provided, using random port in range 31000-32000")
            port = random.randint(31000, 32000)
        if isinstance(gpu_ids, int):
            gpu_ids = [gpu_ids]
        elif isinstance(gpu_ids, str):
            gpu_ids = [int(gpu_id) for gpu_id in gpu_ids.split(",")]
        else:
            gpu_ids = None
        workder_addr, proc = launch_worker_func(
            model_name=model_name,
            num_gpus=num_gpus,
            gpu_ids=gpu_ids,
            dtype=dtype,
            quantization=quantization,
            port=port,
            host=host
        )
        
    def call_worker(
        self,
        model_name: str,
        worker_addr: str,
        prompt: str,
        engine: str="vllm",
        temperature: float=0.0,
        top_p: float=1.0,
        max_tokens: int=None,
        timeout: int=60,
    ) -> str:
        from . import get_call_worker_func
        call_worker_func = get_call_worker_func(
            model_name=model_name,
            worker_addrs=[worker_addr],
            engine=engine,
            use_cache=True,
            overwrite_cache=False
        )
        response = call_worker_func([prompt], temperature=temperature, top_p=top_p, max_tokens=max_tokens, timeout=timeout)
        return response
    
    def clean_workers(
        self,
        engine: str="vllm",
    ):
        print("Note: This will kill all workers for the specified engine")
        if engine == "vllm":
            os.system("pkill -f vllm.entrypoints.openai.api_server")
        elif engine == "sglang":
            os.system("pkill -f sglang.launch_server")
        else:
            raise ValueError(f"Engine {engine} not supported")
        print("Workers cleaned")
        
def main():
    fire.Fire(Cli)
    
if __name__ == "__main__":
    main()
        
        
"""
llm-engines serve "meta-llama/Meta-Llama-3-8B-Instruct" --engine vllm --num-gpus 1 --gpu-ids 2 --dtype auto 
llm-engines call-worker "meta-llama/Meta-Llama-3-8B-Instruct" "http://127.0.0.1:31845" "Hello"
llm-engines clean-workers --engine vllm

"""