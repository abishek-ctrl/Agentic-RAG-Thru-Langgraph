from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import os
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np
from pypdf import PdfReader
from docx import Document
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_CLOUD_URL"), 
    api_key=os.getenv("QDRANT_API_KEY"),
)

collection_name = "documents"
dimension = 384  

# Create collection if it doesn't exist
try:
    qdrant_client.get_collection(collection_name)
except Exception:
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
    )

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Embedding model

# Store document texts for retrieval
documents = []

# Initialize ChatGroq for answer generation
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API"))  # Use GROQ_API environment variable

# Define the state for the LangGraph workflow
class AgentState:
    def __init__(self, files=None, query=None, context=None, answer=None, retrieval_count=3, memory=None):
        self.files = files
        self.query = query
        self.context = context
        self.answer = answer
        self.retrieval_count = retrieval_count  # Number of documents to retrieve
        self.memory = memory if memory is not None else []  # Memory for chat-like flow

# Helper functions
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

# Document Processor Agent
def document_processor_agent(state):
    if state.files is None:
        return state  # Skip if no files are provided
    for file in state.files:
        file_path = f"uploads/{file.filename}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        try:
            if file_path.endswith(".pdf"):
                text = extract_text_from_pdf(file_path)
            elif file_path.endswith(".docx"):
                text = extract_text_from_docx(file_path)
            else:
                raise ValueError("Unsupported file format. Only PDF and Docx are supported.")

            # Generate embeddings
            embedding = model.encode(text).tolist()
            point_id = len(documents)  # Use the length of documents as the point ID
            documents.append(text)  # Store the document text

            # Add embedding to Qdrant
            qdrant_client.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={"text": text}
                    )
                ]
            )
            print("Success appending to docs and Qdrant")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        finally:
            os.remove(file_path)  # Clean up the uploaded file
    print(state)
    return state

# Retriever Agent
def retriever_agent(state):
    query = state.query
    if not query:
        raise ValueError("Query cannot be None.")
    query_embedding = model.encode(query).tolist()
    print("Searching through queries")
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=state.retrieval_count
    )
    print("Got search results.")
    context = "\n".join([hit.payload["text"] for hit in search_result])
    print("Got this context:", context)
    state.context = context
    return state

# Complexity Decider Agent
def complexity_decider_agent(state):
    query = state.query
    prompt = f"""Is the following query complex and requires multi-step reasoning? Answer with 'yes' or 'no'.
    Query: {query}"""
    response = llm.invoke(prompt)
    if "yes" in response.content.lower():
        state.complex = True
    else:
        state.complex = False
    return state

# Query Decomposer Agent
def query_decomposer_agent(state):
    query = state.query
    prompt = f"""Break down the following complex query into simpler sub-queries:
    Query: {query}"""
    response = llm.invoke(prompt)
    sub_queries = response.content.split("\n")
    state.sub_queries = sub_queries
    return state

# Sub-Query Answer Generator Agent
def sub_query_answer_generator_agent(state):
    sub_queries = state.sub_queries
    sub_answers = []
    for sub_query in sub_queries:
        sub_state = AgentState(query=sub_query)
        sub_state = retriever_agent(sub_state)
        sub_state = answer_generator_agent(sub_state)
        sub_answers.append(sub_state.answer)
    state.sub_answers = sub_answers
    return state

# Final Answer Generator Agent
def final_answer_generator_agent(state):
    sub_answers = state.sub_answers
    query = state.query
    
    # Construct the prompt without using a backslash in the f-string
    sub_answers_str = "\n".join(sub_answers)
    prompt = f'''Combine the following sub-answers to answer the original query in a brief and short manner:
    Query: {query}
    Sub-Answers: {sub_answers_str}'''
    
    response = llm.invoke(prompt)
    state.answer = response.content
    return state

# Answer Generator Agent
def answer_generator_agent(state):
    query = state.query
    context = state.context
    combined_input = f"Answer the Query in a brief and short manner. Context: {context}\nQuery: {query}"
    response = llm.invoke(combined_input)
    state.answer = response.content
    return state

# Build the LangGraph workflow
workflow = StateGraph(AgentState)

# Add nodes to the workflow
workflow.add_node("retriever", retriever_agent)
workflow.add_node("answer_generator", answer_generator_agent)
workflow.add_node("complexity_decider", complexity_decider_agent)
workflow.add_node("query_decomposer", query_decomposer_agent)
workflow.add_node("sub_query_answer_generator", sub_query_answer_generator_agent)
workflow.add_node("final_answer_generator", final_answer_generator_agent)

# Define the edges
workflow.add_edge("retriever", "complexity_decider")
workflow.add_conditional_edges(
    "complexity_decider",
    lambda state: "query_decomposer" if state.complex else "answer_generator",
)
workflow.add_edge("query_decomposer", "sub_query_answer_generator")
workflow.add_edge("sub_query_answer_generator", "final_answer_generator")
workflow.add_edge("final_answer_generator", END)
workflow.add_edge("answer_generator", END)

# Set the entry point
workflow.set_entry_point("retriever")

# Compile the workflow
app_workflow = workflow.compile()

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

# API endpoints
@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    state = AgentState(files=files)
    document_processor_agent(state)  # Only process documents
    return {"message": "Files processed and stored successfully."}

@app.post("/query/", response_model=QueryResponse)
async def query_system(request: QueryRequest):
    state = AgentState(query=request.query)
    state = app_workflow.invoke(state)
    state.memory.append({"query": request.query, "answer": state.answer})  # Update memory
    return {"answer": state.answer}

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)