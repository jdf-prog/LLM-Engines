from openai import OpenAI
from typing import List

# no image, multi-turn, do not use openai_generate, but can refer to it
def call_worker_openai(messages:List[str], model_name, conv_system_msg=None, **generate_kwargs) -> str:
    # change messages to openai format
    new_messages = []
    if conv_system_msg:
        new_messages.append({"role": "system", "content": conv_system_msg})
    for i, message in enumerate(messages):
        new_messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": message})
    assert new_messages[-1]["role"] == "user", "The last message must be from the user"
    # initialize openai client
    client = OpenAI()
    # call openai
    response = client.chat.completions.create(
        model=model_name,
        messages=new_messages,
        **generate_kwargs,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    from icecream import ic
    ic(call_worker_openai(["Hello"], "gpt-3.5-turbo"))