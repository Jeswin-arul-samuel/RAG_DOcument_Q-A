import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

st.set_page_config(page_title="Chat with your PDF Documents", page_icon=":printer:")
st.title("Chat with your PDF Documents using GROQ :printer:")

# Sidebar for API keys
open_api_key = st.sidebar.text_input("OPENAI API Key", type="password", placeholder="Enter your OpenAI API key")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password", placeholder="Enter your Groq API key")

if not open_api_key:
    st.info("Please enter your OpenAI API key.")
if not groq_api_key:
    st.info("Please enter your Groq API key.")

uploaded_files = st.file_uploader("Upload your PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    directory = "research_papers"

    if not os.path.exists(directory):
        os.makedirs(directory)

    for uploaded_file in uploaded_files:
        file_path = os.path.join(directory, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())  # saves PDF to directory

    st.success(f"All files were uploaded successfully. Click on 'Document Embedding' to Continue.")

llm = ChatGroq(api_key=groq_api_key, model="llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
Answer the questions based on the provided context.
Please provide the most accurate response based in the question
<context>
{context}
<context>
Question:{input}
"""
)

def create_vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=OpenAIEmbeddings(api_key=open_api_key)  ## Embeddings
        st.session_state.loader=PyPDFDirectoryLoader("research_papers")        ## Data ingestion
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)

if st.button("Document Embedding"):
    create_vector_embeddings()
    st.success("vector database is ready")

user_prompt = st.text_input("Enter your query from the research papers")

import time

if user_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({"input":user_prompt})
    print(f"Response time:{time.process_time()-start}")

    st.write(response["answer"])


## With streamlit expander

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("------------------")