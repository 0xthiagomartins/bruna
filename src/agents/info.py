from .base import BaseAgent
from langchain_core.prompts import BasePromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import Runnable
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_groq import ChatGroq


store: dict = {}


class AutismAwarenessAgent(BaseAgent):

    prompt: BasePromptTemplate = ChatPromptTemplate.from_messages(
        [
            (
                "You are an informational chatbot dedicated to educating users about autism. "
                "This application aims to provide information about autism diagnosis, behaviors, and how families can create a welcoming environment. "
                "The user's role is assessed as '{user_role}', which could be a family member, educator, or caregiver. "
                "Your goal is to respond with accurate information and practical advice based on the user's interest in '{topic_of_interest}'. "
                "If the user is a family member, provide guidance on creating a nurturing environment. "
                "If the user is an educator, focus on inclusive practices and strategies for understanding behaviors. "
                "If the user is a caregiver, give practical tips on communication and daily care. "
                "The following is the user's current area of interest: {topic_of_interest}. "
                "Here are known challenges related to autism: {common_challenges}. "
                "{contextual_response} "
                "Generate a supportive and informative response based on the following inputs: "
                "User's specific question: '{user_question}'. "
                "User role: {user_role}. "
                "Topic of interest: {topic_of_interest}. "
                "Suggested resources or practices: {suggested_resources}. "
            ),
            MessagesPlaceholder(variable_name="conversation_history"),
            ("human", "{input}"),
        ]
    )

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history: BaseChatMessageHistory = self.__get_session_history(session_id)

    def generate_contextual_response(self, user_role, topic_of_interest):
        contextual_response = ""

        match user_role:
            case "family_member":
                if topic_of_interest == "creating a welcoming environment":
                    contextual_response = "Offer suggestions on setting up predictable routines, sensory-friendly spaces, and using supportive language."
                elif topic_of_interest == "diagnosis":
                    contextual_response = "Provide guidance on early signs, seeking professional help, and understanding the diagnostic process."
                elif topic_of_interest == "support strategies":
                    contextual_response = "Suggest practical ways to encourage communication and reinforce positive behaviors."

            case "educator":
                if topic_of_interest == "inclusive practices":
                    contextual_response = "Recommend strategies for inclusive classroom setups, such as visual aids, clear communication, and sensory breaks."
                elif topic_of_interest == "understanding behaviors":
                    contextual_response = "Provide insights into common behaviors and ways to respond supportively in an educational setting."

            case "caregiver":
                if topic_of_interest == "daily care tips":
                    contextual_response = "Offer advice on routines, communication aids, and ways to manage sensory needs."
                elif topic_of_interest == "behavioral guidance":
                    contextual_response = "Suggest approaches for reinforcing positive behaviors and setting consistent boundaries."

        return contextual_response

    def __get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    def __get_chain(self) -> Runnable:
        return self.prompt | ChatGroq(model="llama-3.2-90b-vision-preview")

    def send(
        self,
        message: str,
        user_role: str,
        topic_of_interest: str,
        user_question: str,
        common_challenges: str,
        suggested_resources: str,
    ) -> AIMessage:
        contextual_response = self.generate_contextual_response(
            user_role, topic_of_interest
        )
        # Executa o chain com as variáveis apropriadas
        with_message_history = RunnableWithMessageHistory(
            self.__get_chain(),
            self.__get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        ai_message: AIMessage = with_message_history.invoke(
            {
                "user_role": user_role,
                "topic_of_interest": topic_of_interest,
                "common_challenges": common_challenges,
                "contextual_response": contextual_response,
                "user_question": user_question,
                "suggested_resources": suggested_resources,
            },
            config={"configurable": {"session_id": self.session_id}},
        )
        message = ai_message.to_json().get("kwargs", {})
        return message
