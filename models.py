from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.chat_models.ollama import ChatOllama
from langchain_chroma import Chroma
from utils import get_loader, get_text_splitter


from config import CHAT_MODEL, EMBEDDING_MODEL, PERSIST_DIRECTORY


def get_llm():
    return ChatOllama(model=CHAT_MODEL)


def get_ollama_embeddings():
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def get_chroma():
    return Chroma(
        persist_directory=PERSIST_DIRECTORY, embedding_function=get_ollama_embeddings()
    )


def save_embeddings():
    loader = get_loader()
    splitter = get_text_splitter()
    documents = loader.load()  # use lazy_load in prod
    docs = splitter.split_documents(documents)
    Chroma.from_documents(
        docs, embedding=get_ollama_embeddings(), persist_directory=PERSIST_DIRECTORY
    )
