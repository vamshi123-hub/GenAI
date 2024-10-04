import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with Ollama"

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system","you are a helpful ai assistant. please respond to the question asked"),
    ("user","Question:{question}")
])
def generate_response(question, engine,temperature, max_tokens):
    llm = Ollama(model = engine)
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    answer = chain.invoke({"question":question})
    return answer

# Streamlit Framework
st.title("Q&A Chatbot with Ollama")

#sidebar Drop down to select various Ollama AI Models
engine = st.sidebar.selectbox("select an Ollama Model",["llama3.1","gemma2:2b"])

#Adjust response parameter
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value = 0.6)
max_tokens = st.sidebar.slider("max_tokens", min_value=50, max_value=300, value = 150)

#main Interface for User Input
st.write("Enter Prompt")
user_input = st.text_input("you:")

if user_input:
    response = generate_response(user_input, engine, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please Provide the Query")
