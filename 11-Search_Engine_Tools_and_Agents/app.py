import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks.streamlit import StreamlitCallbackHandler
load_dotenv()
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)
search = DuckDuckGoSearchRun(name="search")

st.title("LangChain Search Engine")

#Side Bar
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter Your Groq API Key:", type = "password")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role":"assistant", "content" : "How Can I Help You😎"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt:= st.chat_input(placeholder="Enter a prompt"):
    st.session_state.messages.append({"role":"user", "content":prompt})
    st.chat_message("user").write(prompt)
    llm = ChatGroq(groq_api_key = api_key, model_name = "Gemma2-9b-It", streaming=True)
    tools = [search, arxiv, wiki]
    search_agent = initialize_agent(tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, handling_passing_errors = False)
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts = False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role":"assistant", "content" : response})
        st.write(response)



