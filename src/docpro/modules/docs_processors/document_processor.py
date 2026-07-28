from PIL import Image

from docpro.modules import BaseProcessorInterface


class DocProcessor:
    def __init__(self, 
                 doc_layout: BaseProcessorInterface, 
                 ocr: BaseProcessorInterface):
        """
        Initializes the document processor with the specified modules.

        Args:
            doc_layout (docpro.modules.BaseProcessorInterface): The module for document layout analysis.
            ocr (docpro.modules.BaseProcessorInterface): The module for Optical Character Recognition (OCR).
        """

        self.layout_module = doc_layout
        self.ocr_module = ocr


    def layout_analyze(self, 
                       page: Image.Image,
                       target_classes: list[str]|None=None,
                       ignor_classes: list[str]|None=None):

        """
        Analyzes the document page layout and returns bounding boxes.

        Processes the page using the layout module and filters the resulting
        boxes by the specified classes. You can specify either the classes to
        keep (`target_classes`) or the classes to exclude (`ignor_classes`).

        Args:
            page (PIL.Image.Image): The document page image.
            target_classes (list[str] | None, optional): List of classes to keep.
                Defaults to None.
            ignor_classes (list[str] | None, optional): List of classes to exclude.
                Defaults to None.

        Returns:
            list[dict]: A list of dictionaries containing information about the
            filtered bounding boxes.

        Raises:
            ValueError: If both `target_classes` and `ignor_classes` are provided.
        """
        
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
        """
        Recognizes text in a specified region of the document page.

        Crops a part of the page image using the coordinates from the `box`
        dictionary, performs OCR using the provided prompt, and adds the
        recognized text to the resulting dictionary.

        Args:
            page (PIL.Image.Image): The full document page image.
            box (dict): A dictionary containing the region coordinates
                (xmin, ymin, xmax, ymax) and other block data.
            prompt (str): The prompt (instruction) for the OCR model.

        Returns:
            dict: A copy of the original `box` dictionary with an added "text"
            key containing the recognized text.
        """

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


        