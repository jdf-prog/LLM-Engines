from setuptools import setup, find_packages
# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='llm-engines',
    version='0.0.23',
    description='A unified inference engine for large language models (LLMs) including open-source models (VLLM, SGLang, Together) and commercial models (OpenAI, Mistral, Claude).',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Dongfu Jiang',
    author_email='dongfu.jiang@uwaterloo.ca',
    packages=find_packages(),
    url='https://github.com/jdf-prog/LLM-Engines',
    entry_points={"console_scripts": ["llm-engines=llm_engines.cli:main"]},
    install_requires=[
        "fire",
        "openai",
        "google-generativeai",
        "accelerate",
        "transformers",
        "torch",
        "Pillow",
        "torch",
        "tqdm",
        "numpy",
        "requests",
        "sentencepiece",
        "vllm>=0.5.5",
        "together",
        "icecream",
        "prettytable",
        "sglang[all]",
        "mistralai",
        "anthropic>=0.36.0",
        "fastapi<=0.112.2",
        "fireworks-ai"
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