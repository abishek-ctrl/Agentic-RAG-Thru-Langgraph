import streamlit as st

def main():
    st.set_page_config(page_title="PDF Chat", page_icon=":books:")

    st.header("PDF Chat :books:")
    st.text_input("Ask a question about the PDF:")

    with st.sidebar:
        st.subheader("Your PDFs")
        st.file_uploader("Upload your PDFs here")
        st.button("Upload")


    

if __name__ == "__main__":
    main()