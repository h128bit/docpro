import os
import io
import dotenv
import fitz
import logging

from docpro.modules import (
    PDFGLMOCRProcessor, 
    PPDoclayoutModelRuntime, 
    GLMOCRProcessor, 
    GLMOCRVLLMConnector,
    BaseProcessorInterface,
    BlocksToMarkdownFormatter,
    BlockFormatter
    )
from docpro.utils import file_or_bytes_to_iobytes, fitz_to_pil



class DocumentProcessor(BaseProcessorInterface):
    def __init__(self, path_to_config: str|os.PathLike):
        dotenv.load_dotenv(path_to_config)

        self._logger = logging.getLogger(self.__class__.__name__)

        try:
            vllm_model_name = os.environ["DOCPRO_VLLM_MODEL_NAME"]
            vllm_model_url = os.environ["DOCPRO_VLLM_MODEL_URL"]
            layout_model_path = os.environ["DOCPRO_LAYOUT_MODEL_PATH"]
        except KeyError as e:
            raise KeyError(f"In configuration file {path_to_config} was not found {e} environment variable")
        

        layout_model = PPDoclayoutModelRuntime(model_path=layout_model_path)
        ocr_connector = GLMOCRVLLMConnector(
            model=vllm_model_name, 
            url=vllm_model_url)

        self.pdf_processor = PDFGLMOCRProcessor(doc_layout=layout_model,
                                                ocr=ocr_connector)

        self.ocr_processor = GLMOCRProcessor(doc_layout=layout_model,
                                             ocr=ocr_connector)

        
                
        self._logger.info("Document processor was created")


    def process(self, 
                file: bytes|io.BytesIO|str|os.PathLike,
                formatter_type: str="blocks",
                force_ocr: bool=False) -> list[dict|str]:
        
        match formatter_type:
            case "markdown":
                formatter = BlocksToMarkdownFormatter()
            case "blocks":
                formatter = BlockFormatter()
            case _:
                raise ValueError(f"Unsupported formatter type. Expected `markdown` or `blocks`. Got {formatter_type}")
        
        buffer = file_or_bytes_to_iobytes(file)

        processed_pages = []

        self._logger.info("Start process file.")
        with fitz.open(stream=buffer) as document:
            for page in document:
                check_text = page.get_text().strip()
                if force_ocr or not check_text:
                    page = fitz_to_pil(page)
                    blocks = self.ocr_processor.process(page)
                else: 
                    blocks = self.pdf_processor.process(page)
                
                formattd_blocks = formatter.process(blocks)

                processed_pages.append(formattd_blocks)

        return processed_pages

