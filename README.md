# Retrieval-Augmented Generation (RAG) Example

This project demonstrates how to implement Retrieval-Augmented Generation
(RAG) for question-answering tasks, leveraging retrieved context from
documents.

## Project Overview

The RAG example chatbot is designed to answer user queries by retrieving
relevant information from a collection of documents. The project consists
of several components:

### Project Structure

* `main.py`: Contains the main logic for running the RAG-based chatbot.
* `models.py`: Defines functions for loading language models, embeddings,
and Chroma vector databases.
* `utils.py`: Provides utility functions for loading and formatting
documents, and splitting text.
* `config.py` (not provided): Define necessary constants such as
`DIRECTORY_PATH`, `CHAT_MODEL`, `EMBEDDING_MODEL`, and
`PERSIST_DIRECTORY`.

### Dependencies

This project relies on the following libraries:

- `langchain_core`
- `langchain_community`
- `langchain_chroma`
- `langchain_text_splitters`

To ensure compatibility, install these libraries using pip:
```bash
pip install -r requirements.txt
```

## Setup

1. Create a `config.py` file with the following constants:

```typescript
DIRECTORY_PATH = "path/to/your/markdown/files"
CHAT_MODEL = "name_of_your_chat_model"
EMBEDDING_MODEL = "name_of_your_embedding_model"
PERSIST_DIRECTORY = "path/to/persist/directory"
```

2. Place your markdown files in the directory specified by
`DIRECTORY_PATH`.
