import base64
from PIL import Image
from io import BytesIO


def image_to_base64(image: Image.Image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue())
    return encoded.decode("utf-8")