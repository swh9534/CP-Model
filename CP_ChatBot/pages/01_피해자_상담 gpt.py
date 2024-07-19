import os
import time
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output import ChatGenerationChunk, GenerationChunk
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import faiss

# api_key.py
def get_api_key():
    with open("api_key.txt", "r") as file:
        return file.read().strip()
api_key = get_api_key()

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""
    
    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()
        
    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")
        
    def on_llm_new_token(self, token: str, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

llm = ChatOpenAI(
    api_key=api_key,
    temperature=0.7,
    streaming=True,
    callbacks=[ChatCallbackHandler(),]
)

st.set_page_config(
    page_title="성범죄, 마약 상담 GPT",
    page_icon="📜",
)

with st.sidebar:
    category = st.selectbox("Select Category", ["Sex Crime Victims", "Drug Addiction Treatment"])
    files = st.file_uploader("Upload .txt, .pdf, or .docx files", type=["pdf", "txt", "docx"], accept_multiple_files=True)

@st.cache_data(show_spinner="Embedding file...")
def embed_files(files, category):
    all_docs = []
    for file in files:
        file_content = file.read()
        print(file.name)
        file_path = f"./.cache/files/{category}/{file.name}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(file_content)

        # 로더 선택
        if file.name.endswith(".txt"):
            loader = TextLoader(file_path, encoding='UTF-8')
        elif file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            st.error(f"Unsupported file type: {file.name}")
            return None
        
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
        )
        docs = loader.load_and_split(text_splitter=splitter)
        all_docs.extend(docs)

    embedder = OpenAIEmbeddings(api_key=api_key)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{category}")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embedder, cache_dir
    )
    vectorstore = faiss.FAISS.from_documents(all_docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    
    return retriever

def save_message(message, role):
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

st.title("성범죄 피해자 및 마약 중독 치료자 지원 GPT")

def paint_history():
    if "messages" in st.session_state:
        for message in st.session_state["messages"]:
            send_message(message["message"], message["role"], save=False)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
        You are a supportive and empathetic AI designed to provide information and support to victims of sex crimes and individuals undergoing drug addiction treatment.
        Answer the questions using the provided context. 
        
        Your goal is to provide comforting and helpful information, offering resources and guidance where possible.
        
        Context: {context}
    """),
    ("human", "{question}")
])

if files and category:
    retriever = embed_files(files, category)
    if retriever:
        send_message("I'm ready to help you. Please ask any questions you have, and I will do my best to provide the information and support you need.", "ai", save=False)
        paint_history()
        message = st.chat_input("Ask anything about your file...")
        if message:
            send_message(message, "human")
            chain = {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            } | prompt | llm
            
            with st.chat_message("ai"):
                chain.invoke(message)
else:
    st.session_state["messages"] = []
    st.markdown("""
    Welcome!
    
    This chatbot is designed to provide information and support to victims of sex crimes and individuals undergoing drug addiction treatment. Use it to ask questions and get informed.
    
    Upload your files on the sidebar to get started.
""")
