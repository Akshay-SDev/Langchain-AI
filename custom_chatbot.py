import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI()
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an Indian movie director, You know details about movies and your name is Rajmoli-AI"),
    ("user", "{input}")
])
output_parser = StrOutputParser()

chain = prompt | llm | output_parser


# from langchain_core.messages import HumanMessage, AIMessage

# chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
# retriever_chain.invoke({
#     "chat_history": chat_history,
#     "input": "Tell me how"
# })

st.title("Hawk-Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt1 := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt1)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt1})

    data = chain.invoke({"input": prompt1})
    response = f"Echo: {data}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
