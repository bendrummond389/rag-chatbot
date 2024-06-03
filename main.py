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

# Define the chat prompt template using system and human message prompts
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question."
            "{context}"
        ),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)


# Function to generate a response using retrieval-augmented generation (RAG)
def generate_rag_response(question):
    # Initialize the Chroma vector database
    db = get_chroma()
    # Configure the retriever with similarity search and set the number of results to return (k=5)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # Define the RAG chain with context retrieval, formatting, and LLM invocation
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | get_llm()
        | StrOutputParser()
    )

    # Invoke the RAG chain with the input question and get the response
    response = rag_chain.invoke(question)
    return response

# Main function to run the chatbot
def main():
    # Uncomment the line below to save embeddings to the vector database before querying
    # save_embeddings()

    # Define the query question
    query = "what does _adapt_ do in sveltekit?"

    # Generate the RAG response for the query
    res = generate_rag_response(query)
    # Print the response
    print(res)

# Entry point of the script
if __name__ == "__main__":
    main()
