from openai import OpenAI
from typing import List

# no image, multi-turn, do not use openai_generate, but can refer to it
def call_worker_openai(messages:List[str], model_name, timeout:int=60, conv_system_msg=None, **generate_kwargs) -> str:
    # change messages to openai format
    new_messages = []
    if conv_system_msg:
        new_messages.append({"role": "system", "content": conv_system_msg})
    for i, message in enumerate(messages):
        new_messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": message})
    # initialize openai client
    client = OpenAI()
    # call openai
    response = client.chat.completions.create(
        model=model_name,
        messages=new_messages,
        timeout=timeout,
        **generate_kwargs,
    )
    return response.choices[0].message.content

def call_worker_openai_completion(prompt:str, model_name, timeout:int=60, **generate_kwargs) -> str:
    # initialize openai client
    client = OpenAI()
    # call openai
    response = client.completions.create(
        model=model_name,
        prompt=prompt,
        timeout=timeout,
        **generate_kwargs,
    )
    return response.choices[0].text

if __name__ == "__main__":
    from icecream import ic
    ic(call_worker_openai(["Hello"], "gpt-3.5-turbo"))
    ic(call_worker_openai_completion("Hello", "gpt-3.5-turbo-instruct"))