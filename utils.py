from langchain_core.messages import SystemMessage
from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import DIRECTORY_PATH


def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, add_start_index=True
    )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_loader():
    return DirectoryLoader(
        DIRECTORY_PATH, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader
    )
