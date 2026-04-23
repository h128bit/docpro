import numpy as np
from PIL import Image

from docpro.modules import BaseProcessorInterface


class DocProcessor:
    def __init__(self, 
                 doc_layout: BaseProcessorInterface, 
                 ocr: BaseProcessorInterface):
        self.layout_module = doc_layout
        self.ocr_module = ocr


    def layout_analyze(self, 
                       page: Image.Image,
                       target_classes: list[str]|None=None,
                       ignor_classes: list[str]|None=None):
        if target_classes is not None and ignor_classes is not None:
            raise ValueError("Both or one from `target_classes` or `ignor_classes` must be None")

        boxes = self.layout_module.process(page)

        if target_classes:
            boxes = list(filter(lambda x: x["class_name"] in target_classes, boxes))
        elif ignor_classes:
            boxes = list(filter(lambda x: x["class_name"] not in ignor_classes, boxes))

        return boxes
    

    def recognize_part(self, 
                       page: Image.Image, 
                       box: dict, 
                       prompt: str):
        xmin = box["xmin"] 
        ymin = box["ymin"] 
        xmax = box["xmax"] 
        ymax = box["ymax"]

        result = box.copy()

        part_coord = (xmin, ymin, xmax, ymax)
        image_part = page.crop(part_coord)
            
        text = self.ocr_module.process(image_part, prompt)

        result["text"] = text
        
        return result


        