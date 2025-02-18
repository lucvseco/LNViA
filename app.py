import os
import streamlit as st
import logging
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage

logging.basicConfig(level=logging.DEBUG)

# Função para carregar e criar vetor baseado no .txt
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
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

# Função para usar o modelo do Groq com ChatGroq
@st.cache_resource
def load_llm():
    api_key = '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'

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",  # Substitua pelo modelo correto
        api_key=api_key,
        temperature=0.5,
        max_tokens=200,
        timeout=None,
        max_retries=2
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
    def qa_chain(question, vectorstore):
        # Recupera os trechos relevantes
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        relevant_texts = retriever.get_relevant_documents(question)

        # Combina os textos relevantes em um único contexto
        context = "\n".join([doc.page_content for doc in relevant_texts])

        # Criação de mensagens no formato esperado pelo LangChain
        messages = [
            SystemMessage(content="Você é um assistente que ajuda com informações baseadas nos editais fornecidos."),
            HumanMessage(content=f"Contexto:\n{context}\n\nPergunta: {question}")
        ]

        # Chama o modelo LLM para gerar a resposta
        response = llm(messages)

        # Retorna a resposta
        return response.content

    # Inicializa o estado de mensagens da sessão
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Exibe mensagens anteriores (se houver)
    for msg in st.session_state.messages:
        if msg["role"] in ["user", "assistant"]:  # Filtra mensagens inválidas
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Caixa de entrada para perguntas
    if question := st.chat_input("Digite sua pergunta:"):
        # Garante o formato correto da mensagem do usuário
        user_message = {"role": "user", "content": question}
        st.session_state.messages.append(user_message)
        with st.chat_message("user"):
            st.markdown(question)

        # Processa a pergunta e gera a resposta com o ChatGroq
        with st.spinner("Gerando resposta com base nos editais..."):
            response = qa_chain(question, vectorstore)  # Passa o vectorstore como argumento

        # Garante o formato correto da mensagem do assistente
        assistant_message = {"role": "assistant", "content": response}
        st.session_state.messages.append(assistant_message)
        with st.chat_message("assistant"):
            st.markdown(response)


if __name__ == "__main__":
    main()

