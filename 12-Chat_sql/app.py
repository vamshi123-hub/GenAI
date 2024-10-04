import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.callbacks import StreamlitCallbackHandler 
from langchain.agents.agent_types import AgentType
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq

st.set_page_config(page_title = "Langchain Chat With SQL DB")
st.title("🦜🔗 Langchain Chat SQL DB")
LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"
radio_opt = ["Use SQLiite3", "Connect to Mysql"]
select_opt = st.sidebar.radio(label = "Choose the Db", options=radio_opt)
if radio_opt.index(select_opt) == 1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("Mysql Host:")
    mysql_user = st.sidebar.text_input("Mysql User:")
    mysql_pass = st.sidebar.text_input("Mysql Password:", type = "password")
    mysql_db = st.sidebar.text_input("Mysql DB:")
else:
    db_uri = LOCALDB

api_key = st.sidebar.text_input(label = "Groq Api Key:", type = "password")
if not db_uri:
    st.info("Please Enter DB Info and Uri.")

if not api_key:
    st.info("Please Add The Groq Api Key.")

llm = ChatGroq(groq_api_key = api_key, model = "Llama3-8b-8192")
@st.cache_resource(ttl = "1h")
def configure_db(db_uri, mysql_host = None, mysql_user = None, mysql_pass = None, mysql_db = None):
    if db_uri == LOCALDB:
        dbfilepath = (Path(__file__).parent/"Student.db").absolute()
        print(dbfilepath)
        creator = lambda:sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri = True)
        return SQLDatabase(create_engine("sqlite:///", creator = creator))
    elif db_uri == MYSQL:
        if not (mysql_host and mysql_user and mysql_pass and mysql_db):
            st.error("Please Provide All Mysql Connection Details.")
            st.stop()
        return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_pass}@{mysql_host}/{mysql_db}"))
    
if db_uri == MYSQL:
    db = configure_db(db_uri, mysql_host, mysql_user, mysql_pass, mysql_db)
else:
    db = configure_db(db_uri)


#ToolKit
toolkit = SQLDatabaseToolkit(db = db, llm = llm)
agent = create_sql_agent(
    llm,
    toolkit,
    verbose = True,
    handle_parsing_errors=True,
    agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

if "messages" not in st.session_state or st.sidebar.button("Clear Message History"):
    st.session_state["messages"] = [{"role" : "assistant", "content" : "How Can I Help You?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder = "Ask anything from provided DB.")

if user_query:
    st.session_state.messages.append({"role" : "user", "content" : user_query})
    st.chat_message("user").write(user_query)
    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(user_query, callbacks = [streamlit_callback])
        st.session_state.messages.append({"role" : "assistant", "content" : response})
        st.write(response)





