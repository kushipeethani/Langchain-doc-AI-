import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

openai_key = input("Enter your OpenAI API Key: ")
os.environ['OPENAI_API_KEY'] = openai_key

pdf_path = input("Enter PDF file path: ")

loader = PyPDFLoader(pdf_path)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(chunks, embeddings)
retriever = db.as_retriever()

llm = ChatOpenAI(model="gpt-4o-mini")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() == "exit":
        print("Thanks! Closing Doc AI.")
        break
    response = qa_chain(query)
    print("\nAnswer:\n", response["result"])
