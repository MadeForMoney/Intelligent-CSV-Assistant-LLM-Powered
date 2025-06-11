from pandasai.llm.local_llm import LocalLLM
import streamlit as st 
import pandas as pd
from langchain_community.llms import Ollama
from pandasai import SmartDataframe
from langchain_google_genai import ChatGoogleGenerativeAI
import getpass
import os
from dotenv import load_dotenv
load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Updated PandasAI configuration
pandas_ai_config = {
    "llm": llm,
    "enable_cache": True,
    "enforce_privacy": False,  # Disable privacy mode to allow numeric operations
    "save_logs": True,
    "verbose": True,
    "conversational": True,
    "pandas_kwargs": {
        "convert_numeric": True
    }
}

st.title("Data Analysis ChatBot")

uploaded_file = st.file_uploader("Upload a CSV file", type=['csv','xlsx'])

if uploaded_file is not None:
    try:
        # Convert data to numeric where possible during load
        data = pd.read_csv(uploaded_file)
        # Automatically convert numeric columns
        for col in data.columns:
            try:
                data[col] = pd.to_numeric(data[col], errors='ignore')
            except:
                continue
                
        st.write(data.head())

        dfs = SmartDataframe(data, config=pandas_ai_config)

        prompt = st.text_input("Enter your prompt:")

        if st.button("Generate"):
            if prompt:
                with st.spinner('Generating Answer.....'):
                    try:
                        response = dfs.chat(prompt)
                        st.write(response)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.info("Try rephrasing your question or use simpler calculations.")

    except Exception as e:
        st.error(f"Error loading file: {str(e)}")