import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmltemp import css, botTemp, userTemp

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

def get_convo_chain(vector):
    llm = ChatMistralAI()  # Use ChatMistralAI instead of ChatOpenAI
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    convo_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector.as_retriever(),
        memory=memory
    )
    return convo_chain

def handle_inp(inp):
    response=st.session_state.conversation({'question':inp})
    st.session_state.chat_history= response['chat_history']

    for i, msg in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.write(userTemp.replace("{{MSG}}",msg.content),unsafe_allow_html=True)
        else:
            st.write(botTemp.replace("{{MSG}}",msg.content),unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="PDF Chat", page_icon=":books:")

    st.write(css,unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation=None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history=None

    st.header("PDF Chat :books:")
    inp= st.text_input("Ask a question about the PDF:")
    if inp:
        handle_inp(inp)

    with st.sidebar:
        st.subheader("Your PDFs")
        docs=st.file_uploader("Upload your PDFs here",accept_multiple_files=True)
        if st.button("Upload"):
            with st.spinner("Processing"):
                rawtxt=get_text(docs)
                chunks=get_chunks(rawtxt)
                vectors=get_vectorstore(chunks)
                st.session_state.conversation=get_convo_chain(vectors)

if __name__ == "__main__":
    main()