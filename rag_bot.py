from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os


load_dotenv()

# Step 1: Load and split the document
loader = TextLoader("data/sample.txt")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

# Step 2: Embed and store in ChromaDB
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(split_docs, embeddings, persist_directory="chroma_store")
vectorstore.persist()

# Step 3: Build retriever and QA chain
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=retriever)

# Step 4: Ask a question
while True:
    query = input("Ask a question: ")
    if query.lower() in ["exit", "quit"]:
        break
    response = qa_chain.run(query)
    print("Answer:", response)
