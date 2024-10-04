import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

st.set_page_config(page_title="Maths ChatGPT")
st.title("Maths Problems Solver and Data Search Assistant")

groq_api_key = st.sidebar.text_input(label="Groq API Key:", type="password")

if not groq_api_key:
    st.info("Please enter API Key.")
    st.stop()

llm = ChatGroq(model = "Gemma2-9b-It", groq_api_key = groq_api_key)

wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name = "Wikipedia",
    func = wikipedia_wrapper.run,
    description = "Tool for searching the internet to find various information on the topics mentioned"
)

#Initialize Math Tool
math_chain = LLMMathChain.from_llm(llm = llm)
calculator = Tool(
    name = "Calculator",
    func = math_chain.run,
    description = "Tool for answering the math realated questions. Only mathematical expressions need to be provided."
)

prompt = """
    Your an agent task for solving users mathematical questions. Logically arrive at the solution and provide a detailed explanation
    and display it points wise for the question below
    Question : {question}
    Answer: 
"""

prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)

#Combine all the tools into chain
chain = LLMChain(llm = llm, prompt = prompt_template)

reasoning_tool = Tool(
    name = "Reasoning tool",
    func = chain.run,
    description = "Tool for answering logical based and reasoning questions"

)

assistant_agent = initialize_agent(
    tools = [wikipedia_tool, calculator, reasoning_tool],
    llm = llm,
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose = False,
    handle_parsing_errors = True

)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role" : "assistant", "content" : "Hi, Iam Maths Chatbot"
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


question = st.text_area("Enter Your Question", "I have 5 bananas and and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. How many total pieces of fruits do i have at the end?")
submit = st.button("Answer:")
if submit:
    if question:
        with st.spinner("Generating..."):
            st.session_state.messages.append({"role" : "user", "content" : question})
            st.chat_message("user").write(question)
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts = True)
            response = assistant_agent.run(st.session_state.messages, callbacks = [st_cb])
            st.session_state.messages.append({"role" : "assistant", "content" : response})
            st.write("The Response is:")
            st.success(response)
    else:
        st.warning("Please Enter Your question.")
