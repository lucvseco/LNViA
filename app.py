import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
from perguntas import perguntas_prontas

# Função para carregar e criar vetor baseado no .txt
@st.cache_resource
def load_vectorstore(txt_file):
    with open(txt_file, "r", encoding="utf-8") as file:
        full_text = file.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(full_text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

# Função para "carregar"o modelo de llm via Groq
@st.cache_resource
def load_llm():
    api_key = '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'

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key,
        temperature=0.5,
        max_tokens=600,
        timeout=None,
        max_retries=2
    )
    return llm

#Função principal
def main():
    st.title("LNViA")
    st.subheader("Assistente de Editais de Iniciação Científica")
    txt_file = "base_textual.txt"
    with st.spinner("Carregando e processando a base de dados..."):
        vectorstore = load_vectorstore(txt_file)

    llm = load_llm()
    # Criação do fluxo de perguntas e respostas (Q&A)
    def qa_chain(question, vectorstore, message_history):
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        relevant_texts = retriever.get_relevant_documents(question)
        context = "\n".join([doc.page_content for doc in relevant_texts])
        history_context = ""
        for msg in message_history[-5:]:
            if msg["role"] == "user":
                history_context += f"Usuário: {msg['content']}\n"
            elif msg["role"] == "assistant":
                history_context += f"Assistente: {msg['content']}\n"
        messages = [
            SystemMessage(content="Você é um assistente que ajuda com informações baseadas nos editais fornecidos."),
            HumanMessage(
                content=f"Contexto relevante:\n{context}\n\nHistórico de conversa:\n{history_context}\n\nPergunta atual: {question}")
        ]
        response = llm(messages)
        return response.content

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "mostrar_todas" not in st.session_state:
        st.session_state.mostrar_todas = False

    for msg in st.session_state.messages:
        if msg["role"] in ["user", "assistant"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    with st.sidebar:
        st.markdown("###  Perguntas Frequentes:")

        todas_perguntas = list(perguntas_prontas().values())
        perguntas_a_exibir = todas_perguntas if st.session_state.mostrar_todas else todas_perguntas[:5]

        for pergunta in perguntas_a_exibir:
            if st.button(pergunta, key=f"pergunta_{pergunta}"):
                st.session_state.selected_question = pergunta
                st.session_state.update()

        if st.session_state.mostrar_todas:
            if st.button("Ver Menos 🔼"):
                st.session_state.mostrar_todas = False
                st.session_state.update()
        else:
            if len(todas_perguntas) > 5 and st.button("Ver Mais 🔽"):
                st.session_state.mostrar_todas = True
                st.session_state.update()

    question = None

    if "selected_question" in st.session_state:
        question = st.session_state.selected_question
        del st.session_state.selected_question

    manual_question = st.chat_input("Digite sua pergunta:")
    if manual_question:
        question = manual_question

    if question:
        user_message = {"role": "user", "content": question}
        st.session_state.messages.append(user_message)
        with st.chat_message("user"):
            st.markdown(question)

        with st.spinner("Gerando resposta com base nos editais..."):
            response = qa_chain(question, vectorstore, st.session_state.messages)

        assistant_message = {"role": "assistant", "content": response}
        st.session_state.messages.append(assistant_message)
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()