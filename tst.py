import pandas as pd
from pandasai import SmartDataframe
from langchain_google_genai import ChatGoogleGenerativeAI
import getpass
import os

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")



llm =ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,


)



df = pd.read_excel(r"C:\Users\amnit\Downloads\titanic.csv")
df = SmartDataframe(df, config={"llm": llm})

print( df.chat('How many rows are there?'))
#print( df.chat('What is the sum of the GDPs of the 2 happiest countries?'))
