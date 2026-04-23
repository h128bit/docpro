from docpro.modules.interfaces import BaseProcessorInterface


class BlocksToMarkdownFormatter(BaseProcessorInterface):
    def __init__(self):
        super().__init__()

    def process(self, blocks):
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
