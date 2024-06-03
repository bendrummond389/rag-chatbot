# Retrieval-Augmented Generation (RAG) Example

This project is an example of how Retrieval-Augmented Generation (RAG) can be implemented to enhance question-answering tasks using retrieved context from documents.

## Project Structure

- `main.py`: Contains the main logic for running the RAG-based chatbot.
- `models.py`: Defines functions for loading the language model, embeddings, and Chroma vector database.
- `utils.py`: Provides utility functions for loading and formatting documents, and splitting text.
- `config.py`: Configuration file (not provided) that should define necessary constants such as `DIRECTORY_PATH`, `CHAT_MODEL`, `EMBEDDING_MODEL`, and `PERSIST_DIRECTORY`.

## Dependencies

This project relies on the following libraries:

- `langchain_core`
- `langchain_community`
- `langchain_chroma`
- `langchain_text_splitters`

Ensure you have these libraries installed. You can install them using pip:

```bash
pip install -r requirements.txt
```
## Setup

- 1. Create a config.py file with the following constants:
- 2. Place your markdown files in the directory specified by DIRECTORY_PATH.
