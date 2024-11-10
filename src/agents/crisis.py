from .base import BaseAgent
from langchain_core.prompts import BasePromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import Runnable
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_groq import ChatGroq
import os
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings


store: dict = {}


def load_faiss_index(faiss_path):
    """Carrega o índice FAISS a partir do caminho especificado."""
    print(f"Carregando índice FAISS de: {faiss_path}", flush=True)
    embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vector_store = FAISS.load_local(
        faiss_path, embedding_model, allow_dangerous_deserialization=True
    )
    print("Índice FAISS carregado com sucesso.", flush=True)
    return vector_store


def perform_query(vector_store, query_text, top_k=5):
    """Realiza uma consulta de similaridade no índice FAISS."""
    print(f"Realizando consulta: '{query_text}'", flush=True)
    results = vector_store.similarity_search(query_text, k=top_k)
    print(f"Top {top_k} resultados encontrados:", flush=True)
    for idx, result in enumerate(results, start=1):
        print(f"\nResultado {idx}:", flush=True)
        print(f"ID: {result.metadata['id']}", flush=True)
        print(f"Conteúdo: {result.page_content[:500]}...", flush=True)
    return results


class CrisisAgent(BaseAgent):

    prompt: BasePromptTemplate = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Your name is Bruna. You are a compassionate support chatbot for Braine AI."
                "This application is designed to assist individuals with autism during crisis moments. "
                "The crisis level is currently assessed as '{crisis_level}', which ranges from calm to severe. "
                "Your goal is to respond calmly and appropriately based on the level of distress. "
                "In severe cases, consider prompting the user to take deep breaths or try other calming techniques. "
                "The user is a {user_type}, which means they may have different needs and responses during crises. "
                "The following are common triggers that might escalate the user's crisis: {crisis_triggers}. "
                "Generate a supportive message based on the following inputs: "
                "User distress description: '{distress_description}'. "
                "Crisis level: {crisis_level}. "
                "Suggested actions: {suggested_actions}. "
                "Contextual data: {contextual_response} ",
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
        return self.prompt | ChatGroq(model="llama-3.2-90b-vision-preview")

    def send(self, message: str) -> AIMessage:
        mock_crisis_level = "calm"
        mock_user_type = "patient"
        mock_crisis_triggers = ["noise", "crowds"]
        mock_suggested_actions = ["Take deep breaths", "Count to ten"]
        mock_distress_description = "Feeling overwhelmed by the surrounding noise."

        with_message_history = RunnableWithMessageHistory(
            self.__get_chain(),
            self.__get_session_history,
            input_messages_key="input",
            history_messages_key="conversation_history",
        )
        faiss_path = "meu_indice_faiss"  # Caminho para o índice FAISS
        vector_store = load_faiss_index(faiss_path)
        query_text = message
        results = perform_query(vector_store, query_text, top_k=5)
        ai_message: AIMessage = with_message_history.invoke(
            {
                "crisis_level": mock_crisis_level,
                "user_type": mock_user_type,
                "crisis_triggers": mock_crisis_triggers,
                "suggested_actions": mock_suggested_actions,
                "distress_description": mock_distress_description,
                "contextual_response": results,
                "input": message,
            },
            config={"configurable": {"session_id": self.session_id}},
        )
        message = ai_message.to_json().get("kwargs", {})
        return message

    def list_messages(self):
        messages = self.history.messages
        return [message.to_json() for message in messages]
