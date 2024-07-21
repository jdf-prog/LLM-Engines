import os
import time
from .utils import with_timeout
together_client = None
def call_worker_together(messages, model_name, timeout:int=60, conv_system_msg=None, **generate_kwargs) -> str:
    from together import Together
    global together_client
    if together_client is None:
        together_client = Together(api_key=os.environ.get("TOGETHER_API_KEY"), timeout=timeout)

    if model_name.startswith("together_"):
        model_name = model_name.replace("together_", "")
    
    new_messages = []
    if conv_system_msg:
        new_messages.append({"role": "system", "content": conv_system_msg})
    for i, message in enumerate(messages):
        new_messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": message})
    
    @with_timeout(timeout)
    def get_response():
        max_retry_for_unbound_local_error = 10
        retry_count = 0
        while True:
            try:
                response = together_client.chat.completions.create(
                    model=model_name,
                    messages=new_messages,
                    **generate_kwargs,
                )
                break
            except UnboundLocalError as e:
                time.sleep(0.2)
                retry_count += 1
                if retry_count >= max_retry_for_unbound_local_error:
                    
                    raise e
                continue
        return response.choices[0].message.content
    return get_response()

def call_worker_together_completion(prompt:str, model_name, timeout:int=60, **generate_kwargs) -> str:
    from together import Together
    global together_client
    if together_client is None:
        together_client = Together(api_key=os.environ.get("TOGETHER_API_KEY"), timeout=timeout)

    if model_name.startswith("together_"):
        model_name = model_name.replace("together_", "")
    
    @with_timeout(timeout)
    def get_response():
        response = together_client.completions.create(
            model=model_name,
            prompt=prompt,
            **generate_kwargs,
        )
        return response.choices[0].text
    return get_response()