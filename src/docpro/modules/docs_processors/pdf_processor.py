import fitz
from PIL import Image

from docpro.utils import fitz_to_pil
from docpro.modules.docs_processors import DocProcessor
from docpro.modules.llms.vllm_connectors import GLMOCRVLLMConnector
from docpro.modules.layouts_analysers.pp_doclayout import PPDoclayoutModelRuntime



class PDFGLMOCRProcessor(DocProcessor):
    def __int__(self, 
                doc_layout: PPDoclayoutModelRuntime, 
                ocr: GLMOCRVLLMConnector):
        super().__init__(doc_layout, ocr)

    
    def get_text_lines(self, page, clip_box) -> str:
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
        return list with dicts:
        {
        class_id,
        class_name,
        confidence,
        reader_order,
        text,
        xmax,
        xmin,
        ymax,
        ymin
        }
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
        

        
        