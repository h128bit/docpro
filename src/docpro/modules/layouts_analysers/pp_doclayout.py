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
        Initializes the PPDoclayoutModelRuntime for document layout analysis.

        This class is designed to work with the PP-DocLayoutV3-ONNX model 
        (https://huggingface.co/alex-dinh/PP-DocLayoutV3-ONNX).

        Args:
            model_path (str | os.PathLike): Path to the ONNX model file.
            confidence (float, optional): Confidence threshold for filtering detections. 
                Defaults to 0.5.
            image_size (tuple[int, int] | None, optional): Target size (width, height) 
                for resizing the input image. Defaults to (800, 800).
            mean (tuple[int, int, int] | None, optional): Mean values for image 
                normalization. Defaults to [0.485, 0.456, 0.406].
            std (tuple[int, int, int] | None, optional): Standard deviation values 
                for image normalization. Defaults to [0.229, 0.224, 0.225].
            normalize_coef (int, optional): Coefficient for normalizing pixel values. 
                Defaults to 255.
            classes (list[str] | None, optional): List of class names corresponding 
                to the model's output indices. Defaults to a predefined list of 25 
                document layout classes.
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
        """
        Preprocesses the input image for the ONNX model.

        Resizes the image to the target size, normalizes pixel values using 
        the specified mean, standard deviation, and normalization coefficient, 
        and transposes the dimensions to match the model's expected input format.

        Args:
            image (np.ndarray): The input image as a NumPy array.

        Returns:
            tuple[np.ndarray, float, float]: A tuple containing:
                - input_blob (np.ndarray): The preprocessed image tensor ready 
                  for the ONNX model.
                - scale_h (float): The scaling factor applied to the height.
                - scale_w (float): The scaling factor applied to the width.
        """
        
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
        """
        Postprocesses the raw model output into a list of bounding box dictionaries.

        Filters the detected boxes based on the confidence threshold and sorts 
        them by reading order.

        Args:
            boxes (np.ndarray): The raw output array from the ONNX model with 
                shape (N, 7), where N is the number of detections. The values 
                represent [label_index, score, xmin, ymin, xmax, ymax, read_order].

        Returns:
            list[dict]: A list of dictionaries, each representing a detected 
            bounding box with the following keys:
                - class_id (int): The integer ID of the detected class.
                - class_name (str): The string name of the detected class.
                - confidence (float): The confidence score of the detection.
                - read_order (float): The predicted reading order of the element.
                - xmin (float): The minimum x-coordinate of the bounding box.
                - ymin (float): The minimum y-coordinate of the bounding box.
                - xmax (float): The maximum x-coordinate of the bounding box.
                - ymax (float): The maximum y-coordinate of the bounding box.
        """

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
        """
        Processes an input image to detect document layout elements.

        Converts the PIL image to a NumPy array, preprocesses it, runs inference 
        using the ONNX model, and postprocesses the results into a structured 
        list of bounding boxes.

        Args:
            image (PIL.Image.Image): The input document page image.

        Returns:
            list[dict]: A list of dictionaries representing the detected layout 
            elements, sorted by reading order and filtered by confidence.
        """
        
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
