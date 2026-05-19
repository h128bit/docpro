from docpro.modules.interfaces import BaseProcessorInterface



class BlocksToMarkdownFormatter(BaseProcessorInterface):
    def __init__(self):
        super().__init__()

    def process(self, blocks: list[dict]) -> str:
        """
        Process list of dicts as  
        {
            class_id: numpy int,
            class_name: str,
            confidence: numpy float,
            read_order: numpy int,
            text: str,
            xmax: numpy float,
            xmin: numpy float,
            ymax: numpy float,
            ymin: numpy float
        }

        Return: str
        Join strings from input blocks sorted when by `read_order`
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
        super().__init__()
    
    def process(self, blocks: list[dict]) -> list[dict]:
        """
        Process list of dicts as  
        {
            class_id: numpy int,
            class_name: str,
            confidence: numpy float,
            read_order: numpy int,
            text: str,
            xmax: numpy float,
            xmin: numpy float,
            ymax: numpy float,
            ymin: numpy float
        }

        Return: list of dicts as
        {
            class_id: int,
            class_name: str,
            confidence: float,
            read_order: int,
            text: str,
            box: {
                xmax: float,
                xmin: float,
                ymax: float,
                ymin: float
                }
        }

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
