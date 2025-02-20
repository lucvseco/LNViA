import os
from app import load_vectorstore, load_llm  # Importando funções do app.py
from perguntas import perguntas_prontas
from langchain.schema import SystemMessage, HumanMessage

def qa_chain(question, vectorstore, llm):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    relevant_texts = retriever.invoke(question)

    context = "\n".join([doc.page_content for doc in relevant_texts])

    messages = [
        SystemMessage(content="Você é um assistente que ajuda com informações baseadas nos editais fornecidos."),
        HumanMessage(content=f"Contexto:\n{context}\n\nPergunta: {question}")
    ]
    response = llm.invoke(messages)
    return response.content

def salvar_respostas(respostas, arquivo="respostas.txt"):
    with open(arquivo, "w", encoding="utf-8") as file:
        for pergunta, resposta in respostas.items():
            file.write(f"Pergunta: {pergunta}\n")
            file.write(f"Resposta: {resposta}\n")
            file.write("-" * 50 + "\n\n")

def processar_perguntas():
    txt_file = "base_textual.txt"
    vectorstore = load_vectorstore(txt_file)
    llm = load_llm()

    perguntas = perguntas_prontas()
    respostas = {}

    for pergunta in perguntas.values():
        resposta = qa_chain(pergunta, vectorstore, llm)
        respostas[pergunta] = resposta

    salvar_respostas(respostas)
    print("Respostas salvas com sucesso no respostas.txt")

if __name__ == "__main__":
    processar_perguntas()
