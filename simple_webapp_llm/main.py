from fastapi import FastAPI, UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

from end_to_end_llm.src.my_gpt4all import MyGPT4ALL

app = FastAPI()

CHROMA_DB_DIR = "chroma_store"
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

# Load PDF and prepare documents
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

# Embed and store in ChromaDB
def store_docs(docs):
    embedder = HuggingFaceEmbeddings()
    vectorstore = Chroma.from_documents(
        docs,
        embedding=embedder,
        persist_directory=CHROMA_DB_DIR,
    )

    return vectorstore

# Load vectorstore
def get_vectorstore():
    embedder = HuggingFaceEmbeddings()
    return Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embedder
    )

@app.post("/upload/")
async def upload_pdf(file: UploadFile):
    path = f"/tmp/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())

    docs = process_pdf(path)
    store_docs(docs)
    return {"status": "uploaded and indexed"}


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


GPT4ALL_MODEL_FOLDER_PATH= f"{BASE_DIR}/end_to_end_llm/models"

GPT4ALL_MODEL_NAME='Meta-Llama-3-8B-Instruct.Q4_0.gguf'
GPT4ALL_BACKEND='llama'
GPT4ALL_ALLOW_STREAMING=True
GPT4ALL_ALLOW_DOWNLOAD=True


@app.get("/ask/")
async def ask(query: str):
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever()

    llm = MyGPT4ALL(
        model_folder_path=GPT4ALL_MODEL_FOLDER_PATH,
        model_name=GPT4ALL_MODEL_NAME,
        allow_download=GPT4ALL_ALLOW_DOWNLOAD
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    return qa.invoke(query)



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
