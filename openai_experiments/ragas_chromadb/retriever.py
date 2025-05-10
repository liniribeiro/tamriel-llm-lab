"""
Collect and load the data

> Download the data
> chunk the data
> embed and store the chunks
"""
import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from openai_experiments.base import OpenAiRAG

CHROMA_DB_DIR = "chroma_store"
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

# Embed and store in ChromaDB
def store_docs(rag: OpenAiRAG, chunks):
    vectorstore = Chroma.from_documents(
        chunks,
        embedding=rag.embeddings,
        persist_directory=CHROMA_DB_DIR,
    )
    return vectorstore

# Load vectorstore
def get_vectorstore(rag: OpenAiRAG):
    return Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=rag.embeddings
    )

def chunk_and_embed_data(rag: OpenAiRAG, file_name):
    loader = TextLoader(file_name)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    store_docs(rag, chunks)

