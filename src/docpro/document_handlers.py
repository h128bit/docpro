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
        """
        Initializes the DocumentProcessor with configuration from a file.

        Loads environment variables from the specified configuration file to 
        initialize the layout model and OCR connector. Sets up both PDF and 
        image-based OCR processors.

        Args:
            path_to_config (str | os.PathLike): Path to the configuration file 
                (e.g., .env) containing required environment variables 
                (DOCPRO_VLLM_MODEL_NAME, DOCPRO_VLLM_MODEL_URL, DOCPRO_LAYOUT_MODEL_PATH).

        Raises:
            KeyError: If any of the required environment variables are missing 
                from the configuration file.
        Example: 
            >>> from docpro import DocumentProcessor
            >>> from pprint import pprint
            >>> 
            >>> '''
            >>> config.env
            >>> DOCPRO_VLLM_MODEL_NAME = hosted_vllm/glm-ocr
            >>> DOCPRO_VLLM_MODEL_URL = http:127.0.0.1/v1/chat/completions
            >>> 
            >>> DOCPRO_LAYOUT_MODEL_PATH = path_to\PP-DocLayoutV3.onnx  # https://huggingface.co/alex-dinh/PP-DocLayoutV3-ONNX
            >>> '''
            >>> 
            >>> CONFIG_PATH = "config.env"
            >>> PATH_TO_FILE = "file.pdf"
            >>> 
            >>> doc_proc = DocumentProcessor(CONFIG_PATH)
            >>> 
            >>> result = doc_proc.process(PATH_TO_FILE, force_ocr=True)
            >>> 
            >>> pprint(result)
        """

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
        """
        Processes a document file and extracts formatted content from its pages.

        Reads the input file, determines whether to use PDF-native text extraction 
        or force OCR based on the `force_ocr` flag and text presence, and formats 
        the extracted blocks using the specified formatter.

        Args:
            file (bytes | io.BytesIO | str | os.PathLike): The input document file 
                (e.g., PDF) as bytes, a BytesIO object, or a file path.
            formatter_type (str, optional): The type of formatting to apply to the 
                extracted blocks. Must be either "markdown" or "blocks". 
                Defaults to "blocks".
            force_ocr (bool, optional): If True, forces OCR processing for all pages 
                even if extractable text is present. Defaults to False.

        Returns:
            list[dict | str]: A list where each element represents a processed page. 
            The element type depends on the `formatter_type`:
                - If "blocks", returns a list of dictionaries containing block 
                  metadata and text, formatted by `docpro.modules.BlockFormatter`.
                - If "markdown", returns a list of strings with concatenated text, 
                  formatted by `docpro.modules.BlocksToMarkdownFormatter`.

        Raises:
            ValueError: If an unsupported `formatter_type` is provided.
        """
        
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

