from .base import BaseAgent
from langchain_core.runnables import Runnable
from langchain_core.prompts import BasePromptTemplate
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from rich import print
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


store = {}


class Chat(BaseAgent):
    prompt: BasePromptTemplate = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're an assistant who's good at {ability}. Respond in 20 words or fewer",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history: BaseChatMessageHistory = self.__get_session_history(session_id)

    def __get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    def __get_chain(self) -> Runnable:
        return self.prompt | ChatGroq(model="llama3-8b-8192")

    def send(self, message: str) -> AIMessage:
        with_message_history = RunnableWithMessageHistory(
            self.__get_chain(),
            self.__get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        ai_message: AIMessage = with_message_history.invoke(
            {"ability": "math", "input": message},
            config={"configurable": {"session_id": self.session_id}},
        )
        message = ai_message.to_json().get("kwargs", {})
        return message

    def list_messages(self):
        messages = self.history.messages
        return [message.to_json for message in messages]
