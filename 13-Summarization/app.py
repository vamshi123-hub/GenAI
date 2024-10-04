import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

st.set_page_config(page_title="Text Summarization")
st.title("🦜🔗 Text Summarization Using Langchain")
st.subheader("Summarization URL:")

#Groq API
with st.sidebar:
    groq_api_key = st.text_input("Groq Api Key", value = "", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")
llm = ChatGroq(model = "Gemma-7b-It", groq_api_key = groq_api_key)

prompt_template = """
    Provide a Summary of following Content in 400 words:
    Context:{text}
"""
prompt = PromptTemplate(template = prompt_template,
                        input_variables=["text"])

submit = st.button("Summarize")
if submit:
    #Validates all inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please Enter Values")
    elif not validators.url(generic_url):
        st.error("Please Enter a Valid URL")
    else:
        try:
            with st.spinner("Processing..."):
            #Loading the Website/Youtube data
                if "youtube.com" in generic_url:
                    loaders = YoutubeLoader.from_youtube_url(generic_url, add_video_info = True)
                else:
                    loaders = UnstructuredURLLoader(urls = [generic_url], ssl_verify = False, 
                                                headers = {"User-Agent" : "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                
                docs = loaders.load()
                #Chain For Summarization
                chain = load_summarize_chain(
                    llm,
                    chain_type = "stuff",
                    prompt = prompt,

                )
                output_summary = chain.run(docs)
                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception:{e}")
