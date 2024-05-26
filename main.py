from config import DIRECTORY_PATH
from utils import format_docs, get_loader, get_text_splitter
from models import get_llm, get_chroma, get_ollama_embeddings, save_embeddings
from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate


from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question."
            "{context}"
        ),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)


def generate_rag_response(question):
    db = get_chroma()
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | get_llm()
        | StrOutputParser()
    )

    response = rag_chain.invoke(question)
    return response


def main():
    # save_embeddings()

    query = "what does _adapt_  do in sveltekit?"

    res = generate_rag_response(query)
    print(res)


if __name__ == "__main__":
    main()
