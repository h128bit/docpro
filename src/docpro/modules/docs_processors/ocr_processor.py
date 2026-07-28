from PIL import Image

from docpro.modules.docs_processors import DocProcessor
from docpro.modules.llms.vllm_connectors import GLMOCRVLLMConnector
from docpro.modules.layouts_analysers.pp_doclayout import PPDoclayoutModelRuntime


class GLMOCRProcessor(DocProcessor):
    def __int__(self, 
                doc_layout: PPDoclayoutModelRuntime, 
                ocr: GLMOCRVLLMConnector):
        """Initializes the GLMOCRProcessor with specific layout and OCR modules.

        Args:
            doc_layout (docpro.modules.layouts_analysers.pp_doclayout.PPDoclayoutModelRuntime): The document layout analysis module.
            ocr (docpro.modules.llms.vllm_connectors.GLMOCRVLLMConnector): The OCR connector module.
        """

        super().__init__(doc_layout, ocr)


    def process(self, 
                page: Image.Image):
        """Processes a document page to extract layout elements and recognize their content.

        Analyzes the page layout, filters out irrelevant classes (such as images and seals),
        and performs OCR on the remaining elements. Tables are recognized as HTML,
        while other text elements are recognized as Markdown.

        Args:
            page (PIL.Image.Image): The document page image to process.

        Returns:
            list[dict]: A list of dictionaries representing the detected elements.
            Each dictionary contains the following keys:
                - class_id (int): The class identifier.
                - class_name (str): The name of the class.
                - confidence (float): The detection confidence score.
                - read_order (int): The reading order of the element.
                - text (str): The recognized text (HTML for tables, Markdown for others).
                - xmin (int | float): The minimum x-coordinate of the bounding box.
                - ymin (int | float): The minimum y-coordinate of the bounding box.
                - xmax (int | float): The maximum x-coordinate of the bounding box.
                - ymax (int | float): The maximum y-coordinate of the bounding box.
        """

        boxes = self.layout_analyze(page, ignor_classes=["header_image", "image", "seal"])

        for i in range(len(boxes)):
            box = boxes[i]
            cls_name = box["class_name"]

            if cls_name == "table":
                task = "Table Recognition: foramted result as html"
            else:
                task = "Text Recognition: formated result as markdown"
            
            text = self.recognize_part(page, box, task)["text"]

            box["text"] = text
        
        return boxes
