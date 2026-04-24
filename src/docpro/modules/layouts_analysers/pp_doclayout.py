import os

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from docpro.modules import BaseProcessorInterface

labels_list = [
    "abstract",
    "algorithm",
    "aside_text",
    "chart",
    "content",
    "display_formula",
    "doc_title",
    "figure_title",
    "footer",
    "footer_image",
    "footnote",
    "formula_number",
    "header",
    "header_image",
    "image",
    "inline_formula",
    "number",
    "paragraph_title",
    "reference",
    "reference_content",
    "seal",
    "table",
    "text",
    "vertical_text",
    "vision_footnote"
  ]


class PPDoclayoutModelRuntime(BaseProcessorInterface):
    def __init__(self, 
                 model_path: str | os.PathLike,
                 confidence: float=0.5,
                 image_size: tuple[int, int]|None=None,
                 mean: tuple[int, int, int]|None=None,
                 std: tuple[int, int, int]|None=None,
                 normalize_coef: int=255,
                 classes: list[str]|None=None
                 ):
        
        """
        For work with https://huggingface.co/alex-dinh/PP-DocLayoutV3-ONNX
        """

        self.model_path = model_path

        self.confidence = confidence
        self.image_size = image_size if image_size else (800, 800)
        self.mean = mean if mean else np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = std if std else np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.norm_coef = normalize_coef
        self.classes = classes if classes else labels_list

        self._model = ort.InferenceSession(self.model_path)


    def preprocess(self, 
                   image: np.ndarray) -> tuple[np.ndarray, float, float]:
        orig_h, orig_w = image.shape[:2]
        target_w, target_h = self.image_size
        scale_h = target_h / orig_h
        scale_w = target_w / orig_w

        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        input_blob = resized.astype(np.float32) / self.norm_coef

        input_blob = (input_blob - self.mean) / self.std

        input_blob = input_blob.transpose(2, 0, 1)[np.newaxis, ...]

        return input_blob, scale_h, scale_w


    def postprocess(self, 
                    boxes: np.ndarray) -> list[dict]:
        boxes = boxes[boxes[:, 1] > self.confidence]
        boxes = boxes[np.argsort(boxes[:, 6])]

        boxes_dicts = []
        for box in boxes:
            cls_id = int(box[0])
            cls_name = self.classes[cls_id]
            d = dict()
            d["class_id"] = cls_id
            d["class_name"] = cls_name
            d["confidence"] = box[1]
            d["read_order"] = box[6]
            d["xmin"] = box[2]
            d["ymin"] = box[3]
            d["xmax"] = box[4]
            d["ymax"] = box[5]
            boxes_dicts.append(d)

        return boxes_dicts


    def process(self, 
                image: Image.Image) -> list[dict]:
        input_names = [i.name for i in self._model.get_inputs()]
        output_names = [o.name for o in self._model.get_outputs()]

        image = np.array(image)
        # orig_h, orig_w = image.shape[:2]
        input_blob, scale_h, scale_w = self.preprocess(image)
        
        preprocess_shape = [np.array(self.image_size, dtype=np.float32)]
        input_feed = {input_names[0]: preprocess_shape,
                    input_names[1]: input_blob,
                    input_names[2]: [[scale_h, scale_w]]}

        # shape=(300, 7), Values are [label_index, score, xmin, ymin, xmax, ymax, read_order]
        output = self._model.run(output_names, input_feed)[0]

        boxes = self.postprocess(output)

        return boxes
