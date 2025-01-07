# Agentic RAG Using LangGraph

## Overview

This project is a Langgraph-based LLM application that allows users to upload multiple documents (PDF and DOCX) and ask questions through a web UI. The LLM leverages an **agentic RAG (Retrieval-Augmented Generation)** workflow to provide accurate and context-aware answers based on the content of the uploaded documents. The system is designed to handle both simple and complex queries, with advanced capabilities for multi-step reasoning and document retrieval.

## Features

The following features are currently available in this project:

- **Document Upload**: Users can upload multiple PDF and DOCX documents through the web UI.
- **Question Answering**: Users can ask questions related to the content of the uploaded documents.
- **Agentic RAG Workflow**: The system uses a multi-agent workflow to process queries, including:
  - **Document Processing**: Extracts and stores text from uploaded documents.
  - **Retrieval**: Retrieves relevant document chunks based on semantic similarity.
  - **Complexity Decider**: Determines if a query requires multi-step reasoning.
  - **Query Decomposition**: Breaks down complex queries into simpler sub-queries.
  - **Answer Generation**: Generates concise and accurate answers using an LLM (ChatGroq with LLaMA-3.3-70b).
- **Memory**: The system retains knowledge of past interactions to provide context-aware responses.
- **Simple Web UI**: A user-friendly web interface for document upload and querying.

## Installation

To install this project, follow these steps:

1. Clone the repository using:
   ```bash
   git clone https://github.com/abishek-ctrl/pdf-chat-langchain.git
   ```
2. Install all dependencies by running:
   ```bash
   pip install -r requirements.txt
   ```
3. Add your API keys by creating a `.env` file:
   ```plaintext
   GROQ_API="your_groq_api_key"
   QDRANT_CLOUD_URL="your_qdrant_cloud_url"
   QDRANT_API_KEY="your_qdrant_api_key"
   ```
4. Run the FastAPI application using:
   ```bash
   uvicorn main:app --reload
   ```

## Usage

Once the project is installed, users can start using it by following these steps:

1. **Upload Documents**: Use the "Upload" button on the web UI to upload one or more PDF or DOCX documents.
2. **Ask Questions**: Enter your question in the "Ask Question" field and submit it.
3. **View Answers**: The system will process your query and provide an answer based on the content of the uploaded documents.

### Example Queries

- Simple Query: "What is the main topic of the document?"
- Complex Query: "Compare the advantages and disadvantages of the methods discussed in the document."

## Future Improvements

I plan to improve this project in the future by implementing the following features:

- **Additional Document Formats**: Support for more document formats (e.g., TXT, PPTX).
- **Deployment**: Deployment to a public platform for wider accessibility.

### Update (2025-01-06)

The project has been updated with an **agentic RAG workflow** to improve query handling and answer generation. Key changes include:

- **Multi-Agent Workflow**: Introduced agents for document processing, retrieval, complexity decision, query decomposition, and answer generation.
- **Qdrant Integration**: Replaced FAISS with Qdrant for efficient vector-based document retrieval.
- **ChatGroq Integration**: Replaced OpenAI with ChatGroq (LLaMA-3.3-70b) for answer generation.

### Update (2024-01-20)

Added Streamlit URL for access: [https://lang-chat.streamlit.app](https://lang-chat.streamlit.app) - Non Functional as of Jan 2025. 
