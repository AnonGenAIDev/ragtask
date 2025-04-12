import os
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone as PineconeClient

os.environ["GOOGLE_API_KEY"] = "AIzaSyBVW91QopcuHda5boGgv5FFmGW2GmZQc68"
# --- Set environment and credentials ---
os.environ['PINECONE_API_KEY'] = 'pcsk_2Ukerd_CY6Xv8wcQFW5Lq5x2h32jgsBExjVt1kkUtTnsHBfTbVnmbuZiQn9gGknARxJmML'

# Initialize embeddings and LLM
parser = StrOutputParser()
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-thinking-exp-01-21",
    temperature=0,
)

# --- Pinecone setup ---
pc = PineconeClient()
index_name = "ragtask"
index = pc.Index(index_name)

vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# --- Prompt template ---
template = """
You are an expert data insights assistant for a financial institution. Your goal is to help the bank improve the effectiveness of its future marketing campaigns and to analyze what patterns led to successful deposits.

Your response should:
- Identify key indicators and customer characteristics that contribute to successful deposits.
- Highlight patterns or correlations from the dataset.
- Recommend strategies for future marketing efforts based on the analysis.

Avoid making assumptions from the internet or unrelated source of information. Be concise, analytical, and results-focused.

Context: {context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# --- Build retrieval chain ---
chain = (
    {"context": vectorstore.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)

# --- Streamlit UI ---
st.set_page_config(page_title="Bank Marketing Chatbot", page_icon="💡")
st.title("💬 Bank Marketing Assistant")

query = st.text_input("Ask a question about the data:")

if query:
    with st.spinner("Analyzing..."):
        response = chain.invoke(query)
        st.markdown(f"**Answer:** {response}")
