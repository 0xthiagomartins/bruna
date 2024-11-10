from nameko.rpc import rpc
from dotenv import load_dotenv
from src.agents.crisis import CrisisAgent, AIMessage
from nameko.dependency_providers import DependencyProvider
from . import db


class SessionDataDependency(DependencyProvider):

    def get_dependency(self, worker_ctx):
        try:
            session_data = worker_ctx.data["session_data"]
        except KeyError:
            session_data = None
        except AttributeError:
            session_data = None
        return session_data


print(f'Load .env: {load_dotenv(dotenv_path="./resources/.env")}', flush=True)


class BrunaService:
    name = "bruna"
    session_data: dict = SessionDataDependency()

    @rpc
    def send_crisis(self, session_id: str, message: str) -> AIMessage:
        chat = CrisisAgent(session_id)
        message = chat.send(message)
        return message

    @rpc
    def list_messages(self, session_id: str) -> list:
        chat = CrisisAgent(session_id)
        messages = chat.list_messages(session_id)
        return messages
