# CNC Troubleshooting Assistant

## Purpose
This project provides an AI-powered chatbot designed to assist with troubleshooting CNC (Computer Numerical Control) machines. It leverages a knowledge base extracted from a PDF document to answer user queries related to CNC machine issues and solutions.

## Features
- **Document Loading & Chunking**: Utilizes `PDFPlumberLoader` to ingest PDF documents and `SemanticChunker` with `HuggingFaceEmbeddings` for intelligent text segmentation.
- **Vector Store**: Employs FAISS (Facebook AI Similarity Search) as a vector store to efficiently store and retrieve document embeddings.
- **Language Model**: Integrates with Ollama, using the `gemma3:27b` model, to generate relevant and concise answers based on the retrieved context.
- **Interactive Interface**: Provides a user-friendly chat interface built with Gradio, allowing for real-time interaction with the AI assistant.

## Setup

### Prerequisites
- Python 3.x
- `uv` (a fast Python package installer and resolver)
- Ollama server running with the `gemma3:27b` model. Ensure the Ollama server is accessible at `http://192.168.1.18:11435` as configured in `main.py`.

### Installation
1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd udemy
    ```
2.  **Install dependencies:**
    ```bash
    uv pip install -r requirements.txt
    ```
3.  **Place the knowledge base:**
    Ensure the `cnc_troubleshooting.pdf` file is located in the root directory of the project.

4.  **Start Ollama Server:**
    Make sure your Ollama server is running and the `gemma3:27b` model is available. You might need to pull the model if you haven't already:
    ```bash
    ollama pull gemma3:27b
    ```

### Running the Application
To start the CNC Troubleshooting Assistant, execute the `main.py` script:
```bash
python main.py
```
This will launch a Gradio interface, typically accessible via a local URL (e.g., `http://127.0.0.1:7860`).

## Usage
Once the Gradio interface is running, you can type your questions related to CNC machine troubleshooting into the chat box and receive AI-generated answers based on the provided PDF knowledge base.

## Project Flow
1.  **PDF Document Loading**: The `cnc_troubleshooting.pdf` is loaded using `PDFPlumberLoader`.
2.  **Semantic Chunking**: The document content is split into semantically meaningful chunks using `SemanticChunker` and `HuggingFaceEmbeddings`.
3.  **Vector Store Creation**: These chunks are then embedded and stored in a FAISS vector database.
4.  **Question Answering**: When a user asks a question:
    a.  The question is embedded.
    b.  Relevant document chunks are retrieved from the FAISS vector store based on similarity.
    c.  The retrieved chunks and the user's question are fed to the Ollama `gemma3:27b` model.
    d.  The LLM generates an answer based on the provided context.
5.  **Gradio Interface**: The entire interaction is facilitated through a Gradio web interface, providing a seamless chat experience.