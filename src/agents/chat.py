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
                "You are a compassionate support chatbot for Braine. "
                "This application is designed to assist individuals with autism during crisis moments. "
                "The crisis level is currently assessed as '{crisis_level}', which ranges from calm to severe. "
                "Your goal is to respond calmly and appropriately based on the level of distress. "
                "In severe cases, consider prompting the user to take deep breaths or try other calming techniques. "
                "If the situation escalates, an alert will be sent to {emergency_contact} and {medical_contact}. "
                "The user is a {user_type}, which means they may have different needs and responses during crises. "
                "If the user is a patient, focus on calming and offering reassurance. "
                "If the user is a responsible person, offer guidance on how to support the patient effectively. "
                "The following are common triggers that might escalate the user's crisis: {crisis_triggers}. "
                "{contextual_response} "
                "Generate a supportive message based on the following inputs: "
                "User distress description: '{distress_description}'. "
                "Crisis level: {crisis_level}. "
                "Suggested actions: {suggested_actions}. ",
            ),
            MessagesPlaceholder(variable_name="conversation_history"),
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
        # Mocking config variables
        mock_crisis_level = "calm"
        mock_emergency_contact = "emergency@example.com"
        mock_medical_contact = "medical@example.com"
        mock_user_type = "patient"
        mock_crisis_triggers = ["noise", "crowds"]
        mock_contextual_response = "Please remain calm."
        mock_suggested_actions = ["Take deep breaths", "Count to ten"]
        mock_distress_description = "Feeling overwhelmed by the surrounding noise."

        with_message_history = RunnableWithMessageHistory(
            self.__get_chain(),
            self.__get_session_history,
            input_messages_key="input",
            history_messages_key="conversation_history",
        )
        ai_message: AIMessage = with_message_history.invoke(
            {
                "crisis_level": mock_crisis_level,
                "emergency_contact": mock_emergency_contact,
                "medical_contact": mock_medical_contact,
                "user_type": mock_user_type,
                "crisis_triggers": mock_crisis_triggers,
                "contextual_response": mock_contextual_response,
                "suggested_actions": mock_suggested_actions,
                "distress_description": mock_distress_description,
                "input": message,
            },
            config={"configurable": {"session_id": self.session_id}},
        )
        message = ai_message.to_json().get("kwargs", {})
        return message

    def list_messages(self):
        messages = self.history.messages
        return [message.to_json() for message in messages]
