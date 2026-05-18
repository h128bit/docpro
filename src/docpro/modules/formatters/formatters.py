from docpro.modules.interfaces import BaseProcessorInterface

from pprint import pprint

class BlocksToMarkdownFormatter(BaseProcessorInterface):
    def __init__(self):
        super().__init__()

    def process(self, blocks) -> str:
        result_string = []

        for block in blocks:
            text = block["text"]
            result_string.append(text)

        result_string = "\n".join(result_string)
        return result_string
    

class DammuyFormatter(BaseProcessorInterface):
    def __init__(self):
        super().__init__()
    
    def process(self, blocks):
        return blocks
