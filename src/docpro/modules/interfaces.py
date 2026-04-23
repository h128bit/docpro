from typing import Any
from abc import ABC, abstractmethod


class BaseProcessorInterface(ABC):
    def __init__(self):
        pass 


    @abstractmethod
    def process(self, obj: Any, *args, **kwargs) -> Any:
        raise NotImplementedError