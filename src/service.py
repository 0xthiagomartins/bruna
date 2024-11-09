from nameko.rpc import rpc
from dotenv import load_dotenv
from src.agents.chat import Chat, AIMessage


print(f'Load .env: {load_dotenv(dotenv_path="./resources/.env")}', flush=True)


class BrunaService:
    name = "bruna"

    @rpc
    def send(self, session_id: str, message: str) -> AIMessage:
        chat = Chat(session_id)
        message = chat.send(message)
        return message
