import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader


def get_text(docs):
    text=""
    for pdf in docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def main():
    load_dotenv()
    st.set_page_config(page_title="PDF Chat", page_icon=":books:")

    st.header("PDF Chat :books:")
    st.text_input("Ask a question about the PDF:")

    with st.sidebar:
        st.subheader("Your PDFs")
        docs=st.file_uploader("Upload your PDFs here")
        if st.button("Upload"):
            with st.spinner("Processing"):
                rawtxt=get_text(docs)
                st.write(rawtxt)

if __name__ == "__main__":
    main()