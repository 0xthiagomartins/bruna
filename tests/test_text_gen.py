import pytest
from src.agents.crisis import CrisisAgent
from rich import print


def test_send():
    chat = CrisisAgent("session_id")
    message = chat.send("Me envie uma mensagem de bom dia!")
    print(message)
    assert isinstance(message.get("content"), str)


def test_list_messages():
    chat = CrisisAgent("session_id")
    messages = chat.list_messages()
    print(messages)
