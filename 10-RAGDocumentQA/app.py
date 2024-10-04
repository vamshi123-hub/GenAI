import os
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader



load_dotenv()
#os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key = groq_api_key, model = "Gemma2-9b-It")
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    please provide the most accurate response based on the question

<context>
{context}
<context>
Question:{input}
"""
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model = "gemma2:2b")
        st.session_state.loader = PyPDFDirectoryLoader("Research_Papers")  ##Data Ingestion State
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 150)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
st.title("RAG Document Q&A")
user_prompt = st.text_input("Enter a query from research papers")
if st.button("Document Embeddings"):
    create_vector_embedding()
    st.write("Vector Database is Ready")
import time
if user_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({"input":user_prompt})
    print(f"Response time:{time.process_time()-start}")
    st.write(response['answer'])
    #with a streamlit expander
    with st.expander("Document similarity search"):
        for i,doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("----------------------------------------------")
