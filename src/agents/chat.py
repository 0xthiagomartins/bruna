from .base import BaseAgent
from langchain_core.runnables import Runnable
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from rich import print
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from llama_api_wrapper import LlamaAPI  # Supondo que você tenha uma API para LLaMA

store = {}

class CrisisSupportChat(BaseAgent):
    # Template de prompt para a interação com o usuário em crise
    prompt: BasePromptTemplate = PromptTemplate(
        template=(
            "You are a compassionate support chatbot for Braine. "
            "This application is designed to assist individuals with autism during crisis moments. "
            "The crisis level is currently assessed as '{crisis_level}', which ranges from calm to severe. "
            "Your goal is to respond calmly and appropriately based on the level of distress. "
            "In severe cases, consider prompting the user to take deep breaths or try other calming techniques. "
            "If the situation escalates, an alert will be sent to {emergency_contact} and {medical_contact}. "
            "The user is a {user_type}, which means they may have different needs and responses during crises. "
            "If the user is a patient, focus on calming and offering reassurance. "
            "If the user is a responsible person, offer guidance on how to support the patient effectively. "
            "The historical conversation data for the user is: {conversation_history}. "
            "The following are common triggers that might escalate the user's crisis: {crisis_triggers}. "
            "{contextual_response} "
            "Generate a supportive message based on the following inputs: "
            "User distress description: '{distress_description}'. "
            "Crisis level: {crisis_level}. "
            "Suggested actions: {suggested_actions}. "
        ),
        input_variables=[
            "crisis_level",
            "emergency_contact",
            "medical_contact",
            "user_type",
            "conversation_history",
            "crisis_triggers",
            "contextual_response",
            "distress_description",
            "suggested_actions",
        ],
    )

    def __init__(self, session_id: str, emergency_contact: str, medical_contact: str, model="llama3-8b-8192"):
        self.session_id = session_id
        self.emergency_contact = emergency_contact
        self.medical_contact = medical_contact
        self.history = self.__get_session_history(session_id)
        self.llm = LlamaAPI(model=model)  # Integrando LLaMA através da API
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def __get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    def __get_chain(self) -> Runnable:
        return self.prompt | ChatGroq(model="llama3-8b-8192")

    def generate_contextual_response(self, user_type, crisis_level, conversation_history):
        contextual_response = ""

        if user_type == "paciente":
            contextual_response = (
                "Suggest grounding techniques, deep breathing, or visual calming cues. Avoid overwhelming them with too much information."
            )
        elif user_type == "responsável":
            if crisis_level == "severo":
                contextual_response = (
                    "Advise the responsible person to create a safe, quiet space for the patient. Suggest breathing exercises."
                )
            elif crisis_level == "moderado":
                contextual_response = (
                    "Advise the responsible person to try calming techniques such as offering reassuring words or a comforting object."
                )
            else:  # nível "calmo"
                contextual_response = (
                    "Encourage the responsible person to maintain a peaceful atmosphere and use positive reinforcement."
                )

            if "crises anteriores" in conversation_history.lower():
                contextual_response += (
                    " Reassure the responsible person that their experience will guide the patient through this time."
                )

        return contextual_response

    def send(self, message: str, user_type: str, conversation_history: str, crisis_level: str, distress_description: str, suggested_actions: str, crisis_triggers: str) -> AIMessage:
        # Gera a resposta contextual baseada no tipo de usuário, nível de crise e histórico
        contextual_response = self.generate_contextual_response(user_type, crisis_level, conversation_history)

        # Executa o chain com as variáveis apropriadas
        with_message_history = RunnableWithMessageHistory(
            self.__get_chain(),
            self.__get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        ai_message: AIMessage = with_message_history.invoke(
            {
                "crisis_level": crisis_level,
                "emergency_contact": self.emergency_contact,
                "medical_contact": self.medical_contact,
                "user_type": user_type,
                "conversation_history": conversation_history,
                "crisis_triggers": crisis_triggers,
                "contextual_response": contextual_response,
                "distress_description": distress_description,
                "suggested_actions": suggested_actions,
            },
            config={"configurable": {"session_id": self.session_id}},
        )
        message = ai_message.to_json().get("kwargs", {})
        return message
