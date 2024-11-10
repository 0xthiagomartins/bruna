import pytest
from src.agents.crisis import CrisisAgent
from src.agents.info import AutismAwarenessAgent
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


def test_send():
    agent = AutismAwarenessAgent("session_id2")
    message = agent.send("Tell me something that not everyone knows about autism.")
    print(message)
    assert isinstance(message.get("content"), str)
