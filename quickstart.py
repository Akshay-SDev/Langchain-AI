import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


st.title("chat here...")
input_text = st.text_input("enter text here....")

llm = ChatOpenAI()
# data = llm.invoke(input_text)

prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a goldsmith, who designs necklace. and your name is Shreya."),
    ("user", "{input}")
])
output_parser = StrOutputParser()

chain = prompt | llm | output_parser
data = chain.invoke({"input": input_text})
st.write(data)
