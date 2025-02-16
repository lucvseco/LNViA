import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from translate import Translator
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from wxai_langchain.llm import LangChainInterface
from wxai_langchain.credentials import Credentials
from langchain.embeddings import HuggingFaceEmbeddings

translator = Translator(to_lang="pt")

# Função adaptada para carregar e criar vetor baseado no .txt
@st.cache_resource
def load_vectorstore(txt_file):
    with open(txt_file, "r", encoding="utf-8") as file:
        full_text = file.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(full_text)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Utilizando FAISS para gerenciar o banco vetorial
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore


# Função adaptada para usar o modelo na IBM Cloud
@st.cache_resource
def load_llm():
    # Substitua o dicionário de credenciais por uma instância do objeto Credentials
    creds = Credentials(
        api_key='Z0IsNJbpooE-Yd9Jl_qfcA3Uyo4rXNoyYsY4vNp9lBVZ',
        project_id='c64ffe7c-59b7-4a24-b598-7c2d00de0ffe',
        api_endpoint='https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29'
    )

    # Inicialização do LangChainInterface com as credenciais corrigidas
    llm = LangChainInterface(
        credentials=creds,
        model="meta-llama/llama-2-13b-chat",  # Modelo hospedado no Watsonx
        params={
            "decoding_method": "sample",
            "max_new_tokens": 200,
            "temperature": 0.5,
            "language": "pt"
        }
    )

    return llm

def main():
    st.title("Assistente de Editais de Iniciação Científica")

    # Nome do arquivo base utilizado (seu .txt)
    txt_file = "base_textual.txt"

    # Carrega a base de vetores de texto e o modelo LLM
    with st.spinner("Carregando e processando a base de dados..."):
        vectorstore = load_vectorstore(txt_file)

    llm = load_llm()

    # Criação do fluxo de perguntas e respostas (Q&A)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
    )

    # Inicializa o estado de mensagens da sessão
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Exibe mensagens anteriores (se houver)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Caixa de entrada para perguntas
    if question := st.chat_input("Digite sua pergunta:"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Processa a pergunta e gera a resposta com o Watsonx
        with st.spinner("Gerando resposta com base nos editais..."):
            response = qa_chain.run( question)
            response_pt = translator.translate(response)

        # Armazena e exibe a resposta no chat
        st.session_state.messages.append({"role": "assistant", "content": response_pt})
        with st.chat_message("assistant"):
            st.markdown(response_pt)


if __name__ == "__main__":
    main()
