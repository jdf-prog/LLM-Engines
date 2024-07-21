from setuptools import setup, find_packages
# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='llm-engines',
    version='0.0.2',
    description='A unified inference engine for large language models (LLMs) including open-source models (VLLM, SGLang, Together) and commercial models (OpenAI, Mistral, Claude).',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Dongfu Jiang',
    author_email='dongfu.jiang@uwaterloo.ca',
    packages=find_packages(),
    url='https://github.com/jdf-progLLM-Engines',
    install_requires=[
        "fire",
        "openai",
        "google-generativeai",
        "accelerate",
        "transformers>=4.39.0",
        "torch",
        "Pillow",
        "torch",
        "tqdm",
        "numpy",
        "requests",
        "sentencepiece",
        "vllm==0.5.1", # 0.5.2 does not work, 0.5.1 is the latest version that works, can be updated if newer version is released and it works
        "together",
        "icecream",
        "prettytable",
        "sglang[all]",
        "mistralai",
        "anthropic",
    ],
    extras_require={
        "flash-attn": {
            "flash-attn"
        }
    }
)

"""
rm -rf dist build llm_engines.egg-info
python setup.py sdist bdist_wheel
twine upload dist/*
"""