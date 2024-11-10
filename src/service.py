from nameko.rpc import rpc
from dotenv import load_dotenv
from src.agents.crisis import CrisisAgent, AIMessage


print(f'Load .env: {load_dotenv(dotenv_path="./resources/.env")}', flush=True)


class BrunaService:
    name = "bruna"

    @rpc
    def send_crisis(self, session_id: str, message: str) -> AIMessage:
        chat = CrisisAgent(session_id)
        message = chat.send(message)
        return message

    @rpc
    def list_messages(self, session_id: str) -> list:
        chat = Chat(session_id)
        messages = chat.list_messages(session_id)
        return messages
