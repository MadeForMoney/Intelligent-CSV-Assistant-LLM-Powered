
import streamlit as st 
import pandas as pd
from langchain_community.llms import Ollama
from pandasai import SmartDataframe
from langchain_google_genai import ChatGoogleGenerativeAI
import getpass
import os
from PIL import Image
from dotenv import load_dotenv
load_dotenv()


if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")



llm = ChatGoogleGenerativeAI(
model="models/gemini-1.5-flash",
google_api_key=os.getenv("GOOGLE_API_KEY")
)



st.title("Data Analysis ChatBot ")

uploaded_file=st.file_uploader("Upload a CSV file",type=['csv','xlsx'])

if uploaded_file is not None:
    data=pd.read_csv(uploaded_file)
    st.write(data.head())

    dfs=SmartDataframe(data,config={"llm":llm})

    prompt = st.text_input("Enter your prompt:")

    if st.button("Generate"):
        if prompt:
            with st.spinner('Generating Answer.....'):
                st.write(dfs.chat(prompt))

                image_path = os.path.join('exports', 'charts', 'temp_chart.png')

                # Show image if it exists
                if os.path.exists(image_path):
                    image = Image.open(image_path)
                    st.image(image, caption="Chart", use_column_width=True)



