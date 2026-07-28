# Document Processor

<p align="center">
<img src="images/docpro-logo.png" alt="Document Processor" width="400" height="400">
</p>

<p align="center">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  </a>
  <a href="https://github.com/vllm-project/vllm">
    <img src="https://img.shields.io/badge/vLLM-000000?style=flat-square" alt="vLLM"/>
  </a>
  <a href="https://github.com/BerriAI/litellm">
    <img src="https://img.shields.io/badge/LiteLLM-FF6B35?style=flat-square" alt="LiteLLM"/>
  </a>
  <a href="https://username.github.io/docpro/">
    <img src="https://img.shields.io/badge/Docs-GitHub%20Pages-0A5C36?style=flat-square&logo=github&logoColor=white" alt="Docs"/>
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/OCR_%26_Layout_Analysis-✓-9B59B6?style=flat-square" alt="OCR"/>
  <img src="https://img.shields.io/badge/Table_%26_Form_Extraction-✓-3498DB?style=flat-square" alt="Extraction"/>
  <img src="https://img.shields.io/badge/LLM_Document_Parsing-✓-E67E22?style=flat-square" alt="Parsing"/>
</p>

## About

OCR python package.
Python module for easy embed ocr module in you app.
Module support GLM OCR run on VLLM via LiteLLM.
Also you can build your own configuration using ready-made modules.

## Getting Started

### Insatll

```bash
pip install git+https://github.com/h128bit/docpro
```

### Quick start

```python
from docpro import DocumentProcessor
from pprint import pprint

'''
config.env
DOCPRO_VLLM_MODEL_NAME = hosted_vllm/glm-ocr
DOCPRO_VLLM_MODEL_URL = http:127.0.0.1/v1/chat/completions

DOCPRO_LAYOUT_MODEL_PATH = path_to\PP-DocLayoutV3.onnx
'''

CONFIG_PATH = "config.env"
PATH_TO_FILE = "file.pdf"

doc_proc = DocumentProcessor(CONFIG_PATH)

result = doc_proc.process(PATH_TO_FILE, force_ocr=True)

pprint(result)
```

For more see [documentation](https://h128bit.github.io/docpro/)
