import os
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_pdfs_from_folder(folder_path):
    """Carrega todos os PDFs de uma pasta e extrai o texto."""
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
    return documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", ";", ",", " ", ""],
    )
    documents_chunks = []
    for document in documents:
        document_chunks = text_splitter.split_documents(document)
        documents_chunks += document_chunks
    return documents_chunks


def store_embeddings(chunks, faiss_path):
    embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedding_model.embed_documents(texts)
    metadatas = [{"id": str(uuid.uuid4())} for _ in texts]

    # Armazene no FAISS
    vector_store = FAISS.from_texts(texts, embedding_model, metadatas)
    vector_store.save_local(faiss_path)  # Salva o índice FAISS para consultas futuras
    print(f"Índice FAISS salvo em: {faiss_path}")


def main():
    pdf_folder = "./resources/pdf"  # Pasta com PDFs
    faiss_path = "meu_indice_faiss"  # Caminho para salvar o índice FAISS

    # Carrega e processa os PDFs da pasta
    documents = load_pdfs_from_folder(pdf_folder)
    chunks = split_documents(documents)
    # Armazena embeddings com FAISS
    store_embeddings(chunks, faiss_path)


if __name__ == "__main__":
    main()
