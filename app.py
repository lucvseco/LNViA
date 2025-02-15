import os
import torch
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

@st.cache_resource
def load_vectorstore(txt_file):
    with open(txt_file, "r", encoding="utf-8") as file:
        full_text = file.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Garante que os textos recuperados sejam menores
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(full_text)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

@st.cache_resource
def load_llm():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Troquei para um modelo menor

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        temperature=0.5,  # Respostas mais estáveis e diretas
        top_p=0.8,  # Reduz variação excessiva
        repetition_penalty=1.2,  # Penaliza repetições
        max_new_tokens=200
    )

    llm = HuggingFacePipeline(pipeline=generator)
    return llm

def clean_response(response):
    # Remove o texto indesejado do início e do final da resposta
    start_marker = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."
    end_marker = "Helpful Answer:"
    cleaned_response = response.replace(start_marker, "").replace(end_marker, "").strip()
    return cleaned_response

def main():
    st.title("Assistente de Editais de Iniciação Científica")

    txt_file = "base_textual.txt"

    with st.spinner("Carregando e processando a base de dados..."):
        vectorstore = load_vectorstore(txt_file)

    llm = load_llm()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2})
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
            answer = response.get("result", "").strip()  # Evita espaços extras ou erro de chave
            cleaned_answer = clean_response(answer)  # Limpa a resposta

        st.session_state.messages.append({"role": "assistant", "content": cleaned_answer})
        with st.chat_message("assistant"):
            st.markdown(cleaned_answer)

if __name__ == "__main__":
    main()