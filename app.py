import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

def get_text(docs):
    text= ""
    for pdf in docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_chunks(rawtxt):
    splitter=CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks=splitter.split_text(rawtxt)
    return chunks

def get_vectorstore(chunks):
    embeddings=OpenAIEmbeddings()
    vectors=FAISS.from_texts(texts=chunks,embedding=embeddings)
    return vectors

def main():
    load_dotenv()
    st.set_page_config(page_title="PDF Chat", page_icon=":books:")

    st.header("PDF Chat :books:")
    inp= st.text_input("Ask a question about the PDF:")

    with st.sidebar:
        st.subheader("Your PDFs")
        docs=st.file_uploader("Upload your PDFs here",accept_multiple_files=True)
        if st.button("Upload"):
            with st.spinner("Processing"):
                rawtxt=get_text(docs)
                chunks=get_chunks(rawtxt)
                vectors=get_vectorstore(chunks)

if __name__ == "__main__":
    main()