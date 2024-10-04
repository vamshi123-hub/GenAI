import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system","you are a helpful assistant. please respond to the question asked"),
    ("user","Question:{question}")
])
# Streamlit Framework
st.title("Langchain Demo With Gemma Model")
input_text = st.text_input("Enter Prompt?")
#Ollama Gemma2 Model
llm = Ollama(model = "gemma2:2b")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser
if input_text:
    st.write(chain.invoke({"question":input_text}))
    

