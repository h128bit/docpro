from typing import Any
from abc import ABC, abstractmethod


class BaseProcessorInterface(ABC):
    def __init__(self):
        """Initializes the BaseProcessorInterface."""
        pass 


    @abstractmethod
    def process(self, obj: Any, *args, **kwargs) -> Any:
        """Processes the given object and returns the result.

        This is an abstract method that must be implemented by all subclasses 
        to define their specific processing logic.

        Args:
            obj (Any): The input object to be processed.
            *args: Additional positional arguments for the processing logic.
            **kwargs: Additional keyword arguments for the processing logic.

        Returns:
            Any: The processed result.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError