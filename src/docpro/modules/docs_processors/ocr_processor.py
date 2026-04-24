import fitz 
from PIL import Image

from docpro.modules.docs_processors import DocProcessor
from docpro.modules.llms.vllm_connectors import GLMOCRVLLMConnector
from docpro.modules.layouts_analysers.pp_doclayout import PPDoclayoutModelRuntime


class GLMOCRProcessor(DocProcessor):
    def __int__(self, 
                doc_layout: PPDoclayoutModelRuntime, 
                ocr: GLMOCRVLLMConnector):
        super().__init__(doc_layout, ocr)


    def process(self, 
                page: Image.Image):
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
