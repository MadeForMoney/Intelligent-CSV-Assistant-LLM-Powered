import streamlit as st 
import pandas as pd
from langchain_community.llms import Ollama
from pandasai import SmartDataframe
from pandasai.exceptions import NoCodeFoundError, InvalidOutputValueMismatch, MaliciousQueryError
from langchain_google_genai import ChatGoogleGenerativeAI
import getpass
import os
from PIL import Image
from dotenv import load_dotenv
import time
import builtins
import re
import sys
import io
import collections
import itertools
import functools
import operator
import copy
import pickle
import base64
import hashlib
import warnings
import logging
load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Updated configuration with more permissive settings
pandas_ai_config = {
    "llm": llm,
    "enable_cache": True,
    "enforce_privacy": False,
    "save_logs": True,
    "verbose": True,
    "custom_whitelisted_dependencies": [
        "pandas", 
        "numpy", 
        "sklearn",
        "matplotlib", 
        "seaborn", 
        "scipy",
        "os",
        "pathlib",
        "PIL",
        "tempfile",
        "datetime",
        "json",
        "math",
        "statistics",
        "pandasai",
        "builtins",
        "io",
        "sys",
        "re",
        "collections",
        "itertools",
        "functools",
        "operator",
        "copy",
        "pickle",
        "base64",
        "hashlib",
        "warnings",
        "logging"
    ],
    "max_retries": 3,
    "pandas_kwargs": {
        "convert_numeric": True
    },
    "enable_logging": True,
    "save_charts": True,
    "save_charts_path": "./exports/charts",
    "custom_prompts": {
        "pandas": "You are a helpful data analysis assistant. Use pandas and other data analysis libraries to answer questions about the data."
    }
}

st.title("Data Analysis ChatBot")

uploaded_file = st.file_uploader("Upload a CSV file", type=['csv','xlsx'])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.write(data.head())

        dfs = SmartDataframe(data, config=pandas_ai_config)

        prompt = st.text_input("Enter your prompt:")

        if st.button("Generate"):
            if prompt:
                with st.spinner('Generating Answer.....'):
                    try:
                        # Get the timestamp before processing
                        start_time = time.time()
                        
                        response = dfs.chat(prompt)
                        st.write(response)

                        # Check for charts created after the request started
                        charts_dir = os.path.join('exports', 'charts')
                        if os.path.exists(charts_dir):
                            chart_files = [f for f in os.listdir(charts_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
                            
                            # Find charts that were created after the request started
                            new_charts = []
                            for chart_file in chart_files:
                                chart_path = os.path.join(charts_dir, chart_file)
                                chart_creation_time = os.path.getctime(chart_path)
                                if chart_creation_time >= start_time:
                                    new_charts.append((chart_file, chart_creation_time))
                            
                            if new_charts:
                                # Display the most recent new chart
                                latest_chart = max(new_charts, key=lambda x: x[1])[0]
                                image_path = os.path.join(charts_dir, latest_chart)
                                
                                try:
                                    image = Image.open(image_path)
                                    st.image(image, caption="Generated Chart", use_column_width=True)
                                except Exception as e:
                                    st.warning(f"Could not display chart: {str(e)}")
                            # If no new charts were created, don't show any chart
                                
                    except MaliciousQueryError as e:
                        st.error("❌ The query was blocked due to security restrictions.")
                        st.info("💡 Try rephrasing your question to avoid restricted operations.")
                        st.write(f"Error details: {str(e)}")
                        
                    except InvalidOutputValueMismatch as e:
                        st.error("❌ There was an issue with the output format.")
                        st.info("💡 Try asking for a different type of analysis or visualization.")
                        st.write(f"Error details: {str(e)}")
                        
                    except NoCodeFoundError:
                        st.error("❌ No executable code could be generated from your prompt.")
                        st.info("💡 Try asking questions like:")
                        st.markdown("""
                        - "What is the average of column X?"
                        - "Show me the first 10 rows"
                        - "Create a bar chart of column Y"
                        - "What are the unique values in column Z?"
                        - "Calculate the sum of all numeric columns"
                        - "Show me the correlation between columns A and B"
                        """)
                        
                    except Exception as e:
                        st.error(f"Error processing request: {str(e)}")
                        st.info("Try rephrasing your question or check if your data supports the requested operation.")
                        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        st.info("Please make sure your file is properly formatted and try again.")