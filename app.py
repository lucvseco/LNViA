import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline

@st.cache_resource
def load_vectorstore(txt_file):
    if not os.path.exists(txt_file):
        st.error(f"Arquivo {txt_file} não foi encontrado!")
        return None
    with open(txt_file, "r", encoding="utf-8") as file:
        full_text = file.read()

    # Usa RecursiveCharacterTextSplitter para garantir chunks menores
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(full_text)

    # Validação: divide chunks grandes novamente
    max_chunk_size = 512
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_chunk_size:
            sub_chunks = RecursiveCharacterTextSplitter(
                chunk_size=max_chunk_size,
                chunk_overlap=50,
                separators=["\n\n", "\n", " ", ""]
            ).split_text(chunk)
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(chunk)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(final_chunks, embeddings)
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
    st.title("Sistema RAG para Consultas de Editais de Iniciação Científica")
    st.write("O sistema responde às suas perguntas com base no conteúdo dos PDFs dos editais.")

    txt_file = "base_textual.txt"

    with st.spinner("Carregando e processando a base de dados..."):
        vectorstore = load_vectorstore(txt_file)

    llm = load_llm()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3})  # Limita a 3 chunks
    )

    query = st.text_input("Digite sua pergunta sobre os editais:")
    if st.button("Enviar") and query:
        with st.spinner("Processando sua consulta..."):
            answer = qa_chain.run(query)
        st.markdown("### Resposta:")
        st.write(answer)

if __name__ == "__main__":
    main()
