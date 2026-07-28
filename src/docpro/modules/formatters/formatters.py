from docpro.modules.interfaces import BaseProcessorInterface



class BlocksToMarkdownFormatter(BaseProcessorInterface):
    def __init__(self):
        """Initializes the BlocksToMarkdownFormatter."""

        super().__init__()

    def process(self, blocks: list[dict]) -> str:
        """
        Processes a list of document blocks into a single Markdown string.

        Sorts the input blocks by their reading order and joins their text 
        content with newline characters.

        Args:
            blocks (list[dict]): A list of dictionaries representing document 
                blocks. Each dictionary is expected to contain the following keys:
                - class_id (numpy.int): The class identifier.
                - class_name (str): The name of the class.
                - confidence (numpy.float): The detection confidence score.
                - read_order (numpy.int): The reading order of the block.
                - text (str): The text content of the block.
                - xmax (numpy.float): The maximum x-coordinate of the bounding box.
                - xmin (numpy.float): The minimum x-coordinate of the bounding box.
                - ymax (numpy.float): The maximum y-coordinate of the bounding box.
                - ymin (numpy.float): The minimum y-coordinate of the bounding box.

        Returns:
            str: A single string containing the text from all blocks, 
            sorted by reading order and joined by newlines.
        """

        result_string = []

        blocks.sort(key=lambda x: x["read_order"])
        for block in blocks:
            text = block["text"]
            result_string.append(text)

        result_string = "\n".join(result_string)
        return result_string
    

class BlockFormatter(BaseProcessorInterface):
    def __init__(self):
        """Initializes the BlockFormatter."""

        super().__init__()
    
    def process(self, blocks: list[dict]) -> list[dict]:
        """
        Formats and converts a list of document blocks into a standardized structure.

        Iterates through the input blocks, converts numpy data types to native 
        Python types, and restructures the bounding box coordinates into a 
        nested dictionary.

        Args:
            blocks (list[dict]): A list of dictionaries representing document 
                blocks. Each dictionary is expected to contain the following keys:
                - class_id (numpy.int): The class identifier.
                - class_name (str): The name of the class.
                - confidence (numpy.float): The detection confidence score.
                - read_order (numpy.int): The reading order of the block.
                - text (str): The text content of the block.
                - xmax (numpy.float): The maximum x-coordinate of the bounding box.
                - xmin (numpy.float): The minimum x-coordinate of the bounding box.
                - ymax (numpy.float): The maximum y-coordinate of the bounding box.
                - ymin (numpy.float): The minimum y-coordinate of the bounding box.

        Returns:
            list[dict]: A list of formatted dictionaries. Each dictionary contains:
                - class_id (int): The class identifier.
                - class_name (str): The name of the class.
                - confidence (float): The detection confidence score.
                - read_order (int): The reading order of the block.
                - text (str): The text content of the block.
                - box (dict): A nested dictionary containing the bounding box 
                  coordinates with the following keys:
                    - xmax (float): The maximum x-coordinate.
                    - xmin (float): The minimum x-coordinate.
                    - ymax (float): The maximum y-coordinate.
                    - ymin (float): The minimum y-coordinate.
        """

        results_blocks = []

        for block in blocks:
            d = dict()

            d["class_id"] = block["class_id"]
            d["class_name"] = block["class_name"]
            d["confidence"] = block["confidence"].item()
            d["read_order"] = block["read_order"].item()
            d["text"] = block["text"]
            d["box"] = {
                "xmin": block["xmin"].item(),
                "ymin": block["ymin"].item(),
                "xmax": block["xmax"].item(),
                "ymax": block["ymax"].item()
            }
            results_blocks.append(d)

        return results_blocks
