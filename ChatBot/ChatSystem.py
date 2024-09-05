import streamlit as st
import os
from transformers import AutoTokenizer
from PyPDF2 import PdfReader
from io import BytesIO
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
import re

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Ask me anything."}]



@st.cache_resource(show_spinner=False)
def get_tokenizer():
    """Get a tokenizer to make sure we're not sending too much text
    text to the Model. Eventually we will replace this with ArcticTokenizer
    """
    return AutoTokenizer.from_pretrained("huggyllama/llama-7b")

def get_num_tokens(prompt):
    """Get the number of tokens in a given prompt"""
    tokenizer = get_tokenizer()
    tokens = tokenizer.tokenize(prompt)
    return len(tokens)

def get_pdf_text(pdf_docs): 
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for pages in pdf_reader.pages:
            text += pages.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator ='\n',
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    #embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name ='hkunlp/instructor-xl')
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    return vectorstore

@st.cache_resource(show_spinner=False)
def get_num_tokens(prompt):
    """Get the number of tokens in a given prompt"""
    tokenizer = get_tokenizer()
    tokens = tokenizer.tokenize(prompt)
    return len(tokens)

# Function for generating model response
def generate_response():
    prompt = []
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            prompt.append("<|im_start|>user\n" + dict_message["content"] + "<|im_end|>")
        else:
            prompt.append("<|im_start|>assistant\n" + dict_message["content"] + "<|im_end|>")
    
    prompt.append("<|im_start|>assistant")
    prompt.append("")
    prompt_str = "\n".join(prompt)
    
    if get_num_tokens(prompt_str) >= 3072:
        st.error("Conversation length too long. Please keep it under 3072 tokens.")
        st.button('Clear chat history', on_click=clear_chat_history, key="clear_chat_history")
        st.stop()

    model  = ChatGoogleGenerativeAI(model = 'gemini-pro', temperature = 0.7, top_p = 0.9)
    formatted_input = f"{prompt_str}"
    full_response = ""  # Initialize an empty string to store the response content
    content_pattern = re.compile(r"content='(.*?)'")  # Regular expression to capture content
    
    for event in ChatGoogleGenerativeAI.stream(model, input=formatted_input):
        # Use regular expression to extract content
        matches = content_pattern.findall(str(event))
        for match in matches:
            full_response += match + " "  # Concatenate extracted content
    
    return full_response.strip()  # Return the concatenated response content

def main():
    with st.sidebar:
        st.title('💬 Chatbot')
        if 'HUGGINGFACEHUB_API_TOKEN' and 'GOOGLE_API_KEY' in st.secrets:
            hf_api = st.secrets['HUGGINGFACEHUB_API_TOKEN']
            google_api = st.secrets['GOOGLE_API_KEY']
        else:
            hf_api = st.text_input('Enter Replicate API token:', type='password')
            if not (hf_api.startswith('hf_') and len(hf_api)==37):
                st.warning('Please enter your HuggingFace API token.', icon='⚠️')
                st.markdown("**Don't have an API token?** Head over to [Hugging Face](https://huggingface.co/) to sign up for one.")
        os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_api
        os.environ['GOOGLE_API_KEY'] = google_api

        pdf_docs = st.file_uploader('Add Documents',accept_multiple_files= True)
        if st.button('Process'):
            with st.spinner('Processing'):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.write(text_chunks)
    # Store LLM-generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Ask me anything."}]

    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    st.sidebar.button('Clear chat history', on_click=clear_chat_history)

    # User-provided prompt
    if prompt := st.chat_input(disabled= not hf_api):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

# Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            response = generate_response()
            st.write(response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)

if __name__ == '__main__':
    main()