import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import pipeline

@st.cache_resource
def load_vectorstore(txt_file):
    with open(txt_file, "r", encoding="utf-8") as file:
        full_text = file.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,  # Chunk maior
        chunk_overlap=150,  # Mais sobreposição
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(full_text)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

@st.cache_resource
def load_llm():
    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_length=128,
        truncation=True,
        do_sample=False
    )
    llm = HuggingFacePipeline(pipeline=generator)
    return llm

def main():
    st.title("Assistente de Editais de Iniciação Científica")

    txt_file = "base_textual.txt"

    with st.spinner("Carregando e processando a base de dados..."):
        vectorstore = load_vectorstore(txt_file)

    llm = load_llm()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",  # Estratégia mais avançada
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5})  # Mais trechos
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("Digite sua pergunta:"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.spinner("Gerando resposta com base nos editais..."):
            response = qa_chain.invoke(question)  # Chama o chain
            answer = response['result']  # Mostra apenas o texto da resposta

        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

if __name__ == "__main__":
    main()
