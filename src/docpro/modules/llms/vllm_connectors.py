import litellm 
from PIL import Image
from docpro.utils.image_utils import image_to_base64
from docpro.modules import BaseProcessorInterface


class GLMOCRVLLMConnector(BaseProcessorInterface):
    def __init__(self, 
                 url: str, 
                 model: str="hosted_vllm/glm-ocr"):
        self.url = url
        self.model = model 


    def _get_message(self, 
                     b64_image: str, 
                     task: str|None=None):
        task = task if task else "Text Recognition:"
        messages = {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                        {"type": "text", "text": f"{task}"},
                    ]
                }
        return messages


    def process(self, 
                image: Image.Image, 
                task: str|None=None):
        b64_image = image_to_base64(image)
        messages = [self._get_message(b64_image, task)]

        response = litellm.completion(
            model=self.model, 
            messages=messages,
            api_base=self.url,
            temperature=0.0,
            api_key="your-api-key-here",) 
        
        response = response.to_dict()
        content = response["choices"][0]["message"]["content"]

        return content