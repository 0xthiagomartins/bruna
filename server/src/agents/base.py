from abc import ABC, abstractmethod
from src.config import BaseConfig
from langchain_core.messages import AIMessage


store = {}


class BaseAgent(ABC):
    def __init__(self, config: BaseConfig):
        self.config = config
        self.base_template = "Provide a well-structured and engaging blog post."
        self.references = config.references  # {{ edit_1 }} Initialize references

    @abstractmethod
    def send(self) -> AIMessage:
        pass
