import os
import io
import fitz
from PIL import Image
from pathlib import Path


def file_or_bytes_to_iobytes(object: bytes|io.BytesIO|os.PathLike) -> io.BytesIO:
    if isinstance(object, (str, Path)):
        with open(object, "rb") as f:
            data = f.read(object)
        buffer = io.BytesIO(data)
    elif isinstance(object, bytes):
        buffer = io.BytesIO(object)
    elif isinstance(object, io.BytesIO):
        buffer = object 
    else: 
        raise TypeError(f"Pass was not correct object type. Expected `bytes` or `io.BytesIO` or `PathLike`.\nGot: {type(object)}")
    
    return buffer
    

def fitz_to_pil(object: fitz.Page|Image.Image, dpi=200) -> Image.Image:
    if isinstance(object, fitz.Page):
        pix = object.get_pixmap(dpi=dpi)
        image_page = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    elif isinstance(object, Image.Image):
        image_page = object
    else: 
        raise TypeError(f"Pass was not correct object type. Expected `fitz.Page` or `PIL.Image.Image`.\nGot: {type(object)}")
    return image_page