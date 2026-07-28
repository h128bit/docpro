.. docpro documentation master file, created by
   sphinx-quickstart on Fri Jul 24 16:15:01 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Document processor documentation
================================

OCR python package. 
Python module for easy embed ocr module in you app. 
Module support GLM OCR run on VLLM via LiteLLM. 
Also you can build your own configuration using ready-made modules.



.. code-block:: python
   :caption: example_usage.py

   from docpro import DocumentProcessor
   from pprint import pprint

   """
   config.env
   DOCPRO_VLLM_MODEL_NAME = hosted_vllm/glm-ocr
   DOCPRO_VLLM_MODEL_URL = http:127.0.0.1/v1/chat/completions

   DOCPRO_LAYOUT_MODEL_PATH = path_to\PP-DocLayoutV3.onnx
   """

   CONFIG_PATH = "config.env"
   PATH_TO_FILE = "file.pdf"

   doc_proc = DocumentProcessor(CONFIG_PATH)

   result = doc_proc.process(PATH_TO_FILE, force_ocr=True)

   pprint(result)

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/modules.rst
   api/docpro.modules.docs_processors
   api/docpro.modules.formatters.rst
   api/docpro.modules.layouts_analysers.rst
   api/docpro.modules.llms.rst
   api/docpro.modules.rst
   api/docpro.utils.rst
