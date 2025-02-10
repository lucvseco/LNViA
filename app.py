import os
import streamlit as st
import numpy as np
import  faiss
from langchain.schema import Document

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer
from langchain.storage import InMemoryStore
from langchain_community.embeddings import HuggingFaceEmbeddings

# Verificar se a chave da API do OpenAI está configurada
api_key = st.secrets["OPENAI_API_KEY"]
if not api_key:
    st.error("OpenAI API key not found. Please add it to Streamlit secrets.")
    st.stop()

# Interface do usuário do Streamlit
st.title("LNV ia")
st.write("Ask questions related to open notices from CNPq and FACEPE")

# Definir a pasta local onde os arquivos PDFs estão armazenados; pasta teste
folder_path = r"C:\Users\vitor\Downloads\drive-download-20250209T155226Z-001"   # TODO: Quando for usar substitua pelo caminho da sua pasta

# Valida se a pasta existe
if not os.path.exists(folder_path):
    st.error(f"The folder path '{folder_path}' does not exist. Please check the path and try again.")
else:
    # Lista apenas arquivos PDF na pasta local
    pdf_files = [
        file for file in os.listdir(folder_path)
        if file.endswith('.pdf')  # Verifica extensões .pdf
    ]

    if pdf_files:
        all_docs = []

        with st.spinner("Processing PDFs..."):
            # Processa cada arquivo PDF encontrado
            for file_name in pdf_files:
                file_path = os.path.join(folder_path, file_name)  # Caminho completo do arquivo

                # Utiliza o PyPDFLoader para carregar o conteúdo do PDF
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                all_docs.extend(docs)

            # Exibe o número total de páginas carregadas
            st.success(f"Loaded {len(all_docs)} document pages from {len(pdf_files)} PDF(s).")

        # Dividindo os documentos em textos menores para análise
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(all_docs)

        with st.spinner("Building the vector store..."):
            embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")  # Modelo local
            texts = [doc.page_content.strip() for doc in docs]  # Extraindo o texto de cada documento

            # Verificar se os textos são válidos
            if not texts or not all(isinstance(t, str) for t in texts):
                st.error("Error processing texts. Ensure they are valid strings.")
                st.stop()

            documents = [Document(page_content=text) for text in texts]

            embeddings = [embedding_model.encode(text) for text in texts] # Gerar embeddings localmente
            embeddings = np.array(embeddings, dtype=np.float32)
            embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)#normalização dos embeddings

            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")

            # Verificar se os embeddings têm a forma correta
            if embeddings.ndim != 2 or len(embeddings) != len(texts):
                st.error(f"Embeddings have an unexpected shape: {embeddings.shape}")
                st.stop()

            vectorstore = None

            # # Criar a base de dados FAISS
            # dimension = embeddings.shape[1]  # Dimensão dos embeddings
            # index = faiss.IndexFlatL2(dimension)  # Índice FAISS
            # index.add(embeddings)  # Adiciona os embeddings ao índice

            # Criar um armazenamento de documentos e o mapeamento de índices
            # docstore = InMemoryStore()
            # for i, doc in enumerate(documents):
            #     docstore.mset([(str(i), doc)])
            # index_to_docstore_id = {i: str(i) for i in range(len(documents))}

            # Criar um FAISS armazenável no LangChain
            try:
                vectorstore = FAISS.from_documents(documents, embedding_model)
                st.success("Vector store successfully built!")
            except Exception as e:
                st.error(f"Error during embeddings or vector store creation: {e}")
                st.stop()

        # Validar se `vectorstore` foi criado
        if vectorstore is None:
            st.error("Vectorstore is None. Please review embeddings or process again.")
            st.stop()

        # Definir o retriever
        try:
            retriever = vectorstore.as_retriever()
        except AttributeError:
            st.error("Failed to initialize retriever. Vectorstore might be invalid.")
            st.stop()

        # Usar a API do OpenAI apenas para a geração de respostas
        if api_key:
            llm = OpenAI(temperature=0)
        else:
            st.warning(
                "API Key for OpenAI is not available. Response generation may not work."
            )
            llm = None

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever
        )

        # Interface para perguntas e respostas
        st.markdown("### Ask a Question")
        query = st.text_input("Enter your question here:")

        if query:
            with st.spinner("Generating answer..."):
                answer = qa_chain.run(query)
            st.markdown("*Answer:*")
            st.write(answer)
    else:
        st.info("No PDF files found in the specified folder.")