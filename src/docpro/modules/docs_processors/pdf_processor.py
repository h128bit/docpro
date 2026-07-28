import fitz

from docpro.utils import fitz_to_pil
from docpro.modules.docs_processors import DocProcessor
from docpro.modules.llms.vllm_connectors import GLMOCRVLLMConnector
from docpro.modules.layouts_analysers.pp_doclayout import PPDoclayoutModelRuntime



class PDFGLMOCRProcessor(DocProcessor):
    def __int__(self, 
                doc_layout: PPDoclayoutModelRuntime, 
                ocr: GLMOCRVLLMConnector):
        """
        Initializes the PDF GLM OCR processor with specific layout and OCR modules.

        Args:
            doc_layout (docpro.modules.layouts_analysers.pp_doclayout.PPDoclayoutModelRuntime): The document layout analysis module.
            ocr (docpro.modules.llms.vllm_connectors.GLMOCRVLLMConnector): The OCR connector module.
        """

        super().__init__(doc_layout, ocr)

    
    def get_text_lines(self, page: fitz.Page, clip_box: list) -> str:
        """
        Extracts and concatenates text lines from a specified clipped area of a PDF page.

        Iterates through text blocks, lines, and spans within the given clip box
        to extract and join the text content into a single string.

        Args:
            page (fitz.Page): The PyMuPDF page object to extract text from.
            clip_box (list | tuple): The bounding box coordinates [xmin, ymin, xmax, ymax]
                defining the area to extract text from.

        Returns:
            str: The concatenated text extracted from the specified area.
        """

        text_boxes = page.get_text("dict", 
                                   clip=clip_box, 
                                   flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

        strings = []
        for block in text_boxes:
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    strings.append(text)
        result_text = " ".join(strings)
        return result_text


    def process(self, 
                page: fitz.Page) -> list[dict]:
        
        """
        Processes a PDF page to extract layout elements and recognize their content.

        Converts the PDF page to an image for layout analysis. For detected tables,
        it uses OCR to recognize the content as HTML. For other text elements, it
        scales the bounding boxes and extracts text directly from the PDF using PyMuPDF.

        Args:
            page (fitz.Page): The PyMuPDF page object to process.

        Returns:
            list[dict]: A list of dictionaries representing the detected elements.
            Each dictionary contains the following keys:
                - class_id (int): The class identifier.
                - class_name (str): The name of the class.
                - confidence (float): The detection confidence score.
                - reader_order (int): The reading order of the element.
                - text (str): The extracted text (HTML for tables, plain text for others).
                - xmin (int | float): The minimum x-coordinate of the bounding box.
                - ymin (int | float): The minimum y-coordinate of the bounding box.
                - xmax (int | float): The maximum x-coordinate of the bounding box.
                - ymax (int | float): The maximum y-coordinate of the bounding box.
        """

        image_page = fitz_to_pil(page)
        imw, imh = image_page.size

        boxes = self.layout_analyze(image_page)

        w, h = int(page.rect.width), int(page.rect.height)

        scale_x = w / imw 
        scale_y = h / imh

        for i in range(len(boxes)):
            box = boxes[i]
            if box["class_name"] == "table":
                text = self.recognize_part(image_page, box, "Table Recognition: formatted as html")["text"]

            else:
                clip_box = [
                    box["xmin"] * scale_x,
                    box["ymin"] * scale_y,
                    box["xmax"] * scale_x,
                    box["ymax"] * scale_y,
                ]
                text = self.get_text_lines(page, clip_box)
            box["text"] = text

        return boxes
        

        
        