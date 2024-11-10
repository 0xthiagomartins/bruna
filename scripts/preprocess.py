import os
import uuid
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

print(f'Load .env: {load_dotenv(dotenv_path="./resources/.env")}', flush=True)


def load_pdfs_from_folder(folder_path):
    """Carrega todos os PDFs de uma pasta e extrai o texto."""
    print(f"Iniciando o carregamento de PDFs da pasta: {folder_path}")
    documents = []
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    print(f"Número de arquivos PDF encontrados: {len(pdf_files)}")
    for filename in pdf_files:
        file_path = os.path.join(folder_path, filename)
        print(f"Carregando PDF: {file_path}")
        loader = PyPDFLoader(file_path)
        loaded_docs = loader.load()
        print(f"Documentos carregados do {filename}: {len(loaded_docs)} páginas")
        documents.extend(loaded_docs)
    print(f"Total de documentos carregados: {len(documents)}")
    return documents


def split_documents(documents):
    print("Iniciando a divisão de documentos em chunks.")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", ";", ",", " ", ""],
    )
    # Pass the entire list of documents to the splitter at once
    documents_chunks = text_splitter.split_documents(documents)
    print(f"Total de chunks criados: {len(documents_chunks)}")
    return documents_chunks


def store_embeddings(chunks, faiss_path):
    print("Iniciando o armazenamento de embeddings.")
    embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [{"id": str(uuid.uuid4())} for _ in texts]
    print("Criando o índice FAISS.")

    # Armazene no FAISS
    vector_store = FAISS.from_texts(texts, embedding_model, metadatas)
    vector_store.save_local(faiss_path)  # Salva o índice FAISS para consultas futuras
    print(f"Índice FAISS salvo em: {faiss_path}")


def main():
    print("Iniciando o processo de pré-processamento.")
    pdf_folder = "./resources/pdf"  # Pasta com PDFs
    faiss_path = "meu_indice_faiss"  # Caminho para salvar o índice FAISS
    print(f"Pasta de PDFs: {pdf_folder}")
    print(f"Caminho para o índice FAISS: {faiss_path}")

    # Carrega e processa os PDFs da pasta
    documents = load_pdfs_from_folder(pdf_folder)
    chunks = split_documents(documents)
    # Armazena embeddings com FAISS
    store_embeddings(chunks, faiss_path)
    print("Processo de pré-processamento concluído com sucesso.")


if __name__ == "__main__":
    main()
