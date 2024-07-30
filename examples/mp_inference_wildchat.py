import fire
import json
import os
import datasets
from llm_engines import get_call_worker_func
from llm_engines.utils import MaxRetriesExceededError

def main(
    dataset: str="allenai/WildChat",
    model_name: str="meta-llama/Meta-Llama-3-8B-Instruct",
    engine: str="vllm",
    worker_addrs: str=None,
    num_workers: int=2,
    num_gpu_per_worker: int=1,
    overwrite=False,
    max_size=1000,
):
    # input_file is a hugingface dataset
    dataset = datasets.load_dataset(dataset, split='train')
    if max_size and max_size < len(dataset):
        dataset = dataset.select(range(max_size))
        print(f"Dataset truncated to {max_size} examples")
    
    def format_func(item):
        return {"query": item["conversation"][0]['content']}
    
    dataset = dataset.map(format_func, remove_columns=dataset.column_names)

    output_file="./wildchat.inference.jsonl"
    
    if os.path.exists(output_file) and not overwrite:
        print(f"Output file {output_file} exists and overwrite is set to False. Skipping.")
        exit(0)
    else:
        call_worker = get_call_worker_func(
            model_name=model_name,
            engine=engine,
            worker_addrs=worker_addrs,
            num_workers=num_workers,
            num_gpu_per_worker=num_gpu_per_worker,
            use_cache=True,
            max_retry=1
        )
        
        generation_kwargs = {
            "temperature": 0.0,
            "max_tokens": 4096,
        }
        def map_generate(item):
            try:
                response = call_worker([item['query']], **generation_kwargs)
            except MaxRetriesExceededError:
                response = None
            item['response'] = response
            return item

        
        new_dataset = dataset.map(map_generate, num_proc=num_workers * 16)

        def filter_none(item):
            return item['response'] is not None
        print(f"Before filtering None responses: {len(new_dataset)}")
        new_dataset = new_dataset.filter(filter_none)
        print(f"After filtering None responses: {len(new_dataset)}")

        new_dataset.to_json(output_file, orient="records", lines=True)
    
    
if __name__ == "__main__":
    fire.Fire(main)
    
    
"""
python mp_inference_wildchat.py
"""