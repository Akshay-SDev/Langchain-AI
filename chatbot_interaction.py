from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.retrievers import BaseRetriever


# # First we need a prompt that we can pass into an LLM to generate this search query

# prompt = ChatPromptTemplate.from_messages([
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("user", "{input}"),
# ])

# chat_history = [HumanMessage(content="Hi"), AIMessage(content="Hi!")]
# data = retriever_chain.invoke({
#     "chat_history": chat_history,
#     "input": "what is your name"
# })

# import pdb
# pdb.set_trace()
# print(data['answer'])

llm = ChatOpenAI()
loader = TextLoader("chatbot.txt")
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0
)
docs = loader.load_and_split(
    text_splitter=text_splitter
)
embeddings = OpenAIEmbeddings()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
# chat_history = [HumanMessage(content="Hi!"), AIMessage(content="Hello! How can I assist you today?")]
chat_history = []
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

import streamlit as st

st.title("Hawk-Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if input_text := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(input_text)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": input_text})

    resp = retrieval_chain.invoke({
        "chat_history": chat_history,
        "input": input_text,
        "context": "Chatbot"
    })

    chat_history.append(HumanMessage(content=input_text))
    chat_history.append(AIMessage(content=resp['answer']))
    response = f"Bot: {resp['answer']}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
