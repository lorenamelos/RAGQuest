import os
import openai
import sys

import bs4
import openai
from langchain import hub
from langchain_chroma import Chroma
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage

from langchain_community.document_loaders import BSHTMLLoader
from langchain.chat_models import ChatOpenAI

from langchain_community.document_loaders import UnstructuredURLLoader



# Loading the OpenAI API key:

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)


# Loading HTML with BeautifulSoup4
# loader = BSHTMLLoader("pg55738-images.html")
# docs = loader.load()

# Load HTML documents - in this case, the book

urls = ["https://www.gutenberg.org/cache/epub/55738/pg55738-images.html",
        "https://www.gutenberg.org/cache/epub/42671/pg42671-images.html"
        ]
loader = UnstructuredURLLoader(urls=urls)
docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())


# Retrieve and generate using the relevant snippets of book

retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Adding history
# & Defining a set of questions or prompts that you want to use for retrieval-based question answering


contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""


contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Question-Answer chain

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

chat_history = []

question = "Who is the main author?"

ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})
chat_history.extend([HumanMessage(content=question), ai_msg_1["answer"]])

second_question = "What are the main idea of the the book?"
ai_msg_2 = rag_chain.invoke({"input": second_question, "chat_history": chat_history})

#print(ai_msg_2["answer"])


third_question = "What is the title of the book?"
ai_msg_3 = rag_chain.invoke({"input": third_question, "chat_history": chat_history})

#print(ai_msg_3["answer"])


fourth_question = "Can you summarize in 500 tokens what is the book about?"
ai_msg_4 = rag_chain.invoke({"input": fourth_question, "chat_history": chat_history})

#print(ai_msg_4["answer"])


fifth_question = "Do you know who is Isaac Asimov?"
ai_msg_5 = rag_chain.invoke({"input": fifth_question, "chat_history": chat_history})

#print(ai_msg_5["answer"])

sixth_question = "Do you know who is Jane Austen?"
ai_msg_6 = rag_chain.invoke({"input": sixth_question, "chat_history": chat_history})


seventh_question = "What is the style of the book?"
ai_msg_7 = rag_chain.invoke({"input": seventh_question, "chat_history": chat_history})

#_______________________________________________________________________________________________________________________

# STREAMLIT INTERFACE


import streamlit as st

# Define your RAG chains for each book here (assuming you've already defined them)

# Streamlit UI
st.set_page_config(page_title="RAG", page_icon=":books:")
st.title("RAGQuest: Book Inquiry Engine")


# SIDEBAR SECTION:

# # Text input for OpenAI API key in the sidebar
# openai_api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
# st.sidebar.info("Please enter your OpenAI API key above. This key will be securely stored and used to access OpenAI services.")


# Sidebar for selecting the book
selected_book = st.sidebar.selectbox("Select a Book", ["'The Genetic Effects of Radiation', by Isaac Asimov", "Pride and Prejudice, by Jane Austen"])


# ASK SECTION:

# Define user input field for questions
question = st.text_input(f"Ask a question about '{selected_book}'")

# Define a button to trigger question answering
if st.button("Ask"):
    # Invoke your RAG chain with the user's question and selected number of tokens
    ai_msg = rag_chain.invoke({"input": question, "chat_history": []})

    # Display the answer generated by the RAG model
    st.write("Answer:", ai_msg["answer"])
