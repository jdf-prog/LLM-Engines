import os
together_client = None
def call_worker_together(messages, model_name, conv_system_msg=None, **generate_kwargs) -> str:
    from together import Together
    global together_client
    if together_client is None:
        together_client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

    if model_name.startswith("together_"):
        model_name = model_name.replace("together_", "")
    
    assert len(messages) % 2 == 1, "The number of messages must be odd, meaning the last message is from the user"
    new_messages = []
    for i, message in enumerate(messages):
        new_messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": message})
    assert new_messages[-1]["role"] == "user", "The last message must be from the user"
    
    response = together_client.chat.completions.create(
        model=model_name,
        messages=new_messages,
        **generate_kwargs,
    )
    # print(response.choices[0].message.content)
    return response.choices[0].message.content