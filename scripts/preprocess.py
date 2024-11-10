import uuid
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings


def load_pdfs(pdf_paths):
    documents = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        documents.extend(loader.load())
    return documents


def store_embeddings(documents):
    # Gere embeddings para os documentos
    embeddings = OpenAIEmbeddings()  # Aqui, use o modelo de embeddings desejado

    # Associe cada documento a um UUID
    texts = [doc.text for doc in documents]
    metadatas = [{"id": str(uuid.uuid4())} for _ in texts]

    # Armazene no FAISS
    vector_store = FAISS.from_texts(texts, embeddings, metadatas)
    return vector_store


def main(pdf_paths, faiss_path):
    # Carrega e processa os PDFs
    documents = load_pdfs(pdf_paths)

    # Armazena embeddings com FAISS
    vector_store = store_embeddings(documents)

    # Salva o índice FAISS para consultas futuras
    vector_store.save_local(faiss_path)
    print(f"Índice FAISS salvo em: {faiss_path}")


if __name__ == "__main__":
    pdf_paths = ["exemplo1.pdf", "exemplo2.pdf"]  # Substitua com seus PDFs
    faiss_path = "meu_indice_faiss"
    main(pdf_paths, faiss_path)
