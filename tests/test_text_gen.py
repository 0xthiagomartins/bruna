import pytest
from src.agents.chat import Chat
from rich import print


def test_send():
    chat = Chat("session_id")
    message = chat.send("Me envie uma mensagem de bom dia!")
    print(message)
    assert isinstance(message.get("content"), str)


def test_list_messages():
    chat = Chat("session_id")
