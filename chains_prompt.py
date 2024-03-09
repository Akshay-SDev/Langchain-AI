import os
from langchain.llms import OpenAI
import streamlit as st
from langchain import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

st.title("search about the company")
input_text = st.text_input("enter the company name here...")

llm = OpenAI(temperature=0.8)
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template='tell me about {name} company'
)
chain = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True, output_key='company')

second_input_prompt = PromptTemplate(
    input_variables=['company'],
    template='when was {company} established'
)
chain2 = LLMChain(llm=llm, prompt=second_input_prompt, verbose=True, output_key='founded_year')

third_input_prompt = PromptTemplate(
    input_variables=['founded_year'],
    template='tell me the news on {founded_year}'
)
chain3 = LLMChain(llm=llm, prompt=third_input_prompt, verbose=True, output_key='news')
parent_chain = SequentialChain(chains=[chain, chain2, chain3], input_variables=['name'], output_variables=['company', 'founded_year', 'news'], verbose=True)

if input_text:
    st.write(parent_chain({'name':input_text}))
