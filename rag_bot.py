from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Step 1: Load and split the document
loader = TextLoader("data/sample.txt")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
# Add metadata when splitting documents
split_docs = splitter.split_documents(docs)
for i, doc in enumerate(split_docs):
    doc.metadata["source"] = "sample.txt"
    doc.metadata["chunk_id"] = i


# Step 2: Embed and store in ChromaDB using HuggingFace embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(split_docs, embedding_model, persist_directory="chroma_store")
vectorstore.persist()

# Step 3: Build retriever and QA chain using HuggingFaceHub LLM
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-large",
    task="text2text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    temperature=0.5,
    max_new_tokens=300
)

retriever = vectorstore.as_retriever(search_kwargs={
    "k": 3,
    "filter": {"source": "sample.txt"}  # metadata filter
})

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Step 4: Ask a question
print("📌 Type your question (or 'exit' to quit):")
while True:
    query = input("🧠 You: ")
    if query.lower() in ["exit", "quit"]:
        break
    response = qa_chain.invoke(query)
    print("🤖 Answer:", response)
