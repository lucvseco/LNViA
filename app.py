import os
import tempfile
import streamlit as st

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA


if "OPENAI_API_KEY" not in os.environ:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable or add it to Streamlit secrets.")
    st.stop()


# Streamlit UI

st.title("LLM Chatbot with PDF Retrieval Augmented Generation")
st.write("Upload PDF documents to use as the knowledge base, then ask questions!")


uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_docs = []

    with st.spinner("Processing PDFs..."):

        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name


            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
            all_docs.extend(docs)

        # Optional: Display the number of documents loaded
        st.success(f"Loaded {len(all_docs)} document pages from {len(uploaded_files)} PDF(s).")


    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(all_docs)

    with st.spinner("Building the vector store..."):
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        st.success("Vector store successfully built!")


    retriever = vectorstore.as_retriever()
    llm = OpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


    st.markdown("### Ask a Question")
    query = st.text_input("Enter your question here:")

    if query:
        with st.spinner("Generating answer..."):
            answer = qa_chain.run(query)
        st.markdown("**Answer:**")
        st.write(answer)
else:
    st.info("Please upload one or more PDF files to begin.")
