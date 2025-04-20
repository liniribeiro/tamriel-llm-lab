import os
from typing import Optional

import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_community.embeddings import GPT4AllEmbeddings
from omegaconf import DictConfig


TARGET_SOURCE_CHUNKS=4
CHUNK_SIZE=500
CHUNK_OVERLAP=50
HIDE_SOURCE_DOCUMENTS=False


class MyKnowledgeBase:
    def get_embeddings(self):
        embeddings = GPT4AllEmbeddings()
        return embeddings


    def __init__(self, cfg: DictConfig) -> None:
        """
        Loads pdf and creates a Knowledge base using the Chroma
        vector DB.
        Args:
            pdf_source_folder_path (str): The source folder containing
            all the pdf documents
        """
        self.cfg = cfg
        self.chroma_settings = Settings(**self.cfg.chroma.settings)

    def load_pdfs(self):
        """
        Instantiate the DirectoryLoader class and provide the source document folders inside the constructor.
        """
        # instantiate the DirectoryLoader class
        # load the pdfs using loader.load() function

        loader = DirectoryLoader(
            self.cfg.chroma.document_source_directory
        )
        loaded_pdfs = loader.load()
        return loaded_pdfs

    def split_documents(self, loaded_docs, chunk_size: Optional[int] = 500, chunk_overlap: Optional[int] = 20):
        """
        Split our documents into the some number of chunks where each chunk will have a size of characters.
        :param loaded_docs:
        :param chunk_size:
        :param chunk_overlap: helps to ensure that adjacent chunks share some common characters,
         preventing potential information loss at the boundary between the chunks
        :return: chunked_docs
        """
        # instantiate the RecursiveCharacterTextSplitter class by providing the chunk_size and chunk_overlap
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # Now split the documents into chunks and return
        chunked_docs = splitter.split_documents(loaded_docs)
        return chunked_docs

    def convert_document_to_embeddings(
        self, chunked_docs, embedder
    ):
        """
        Create embeddings for our chunked documents and register those embeddings
        into our knowledge base which will be our vector database
        :param chunked_docs:
        :param embedder: function that will map our chunked documents to embeddings
        :return:
        """

        chroma_db_directory = self.cfg.chroma.db_directory
        if not os.path.exists(chroma_db_directory):
            os.makedirs(chroma_db_directory)

        client = chromadb.Client(self.cfg.chroma.settings)

        # Check if the collection already existstry:
        # Create a collection (or get an existing one)
        collection = client.get_or_create_collection(
            name="knowledge_base",
            embedding_function=embedder
        )

        # Add documents to the collection
        for i, doc in enumerate(chunked_docs):
            collection.add(
                documents=[doc.page_content],
                metadatas=[doc.metadata],
                ids=[f"doc_{i}"]
            )

        print(f"Persist directory: {chroma_db_directory}")
        print(f"Collection exists: {collection.name}")
        print(f"Document count: {collection.count()}")

        return client

    def return_retriever_from_persistant_vector_db(
        self):

        chroma_db_directory = self.cfg.chroma.db_directory

        if not os.path.isdir(chroma_db_directory):
            raise NotADirectoryError("Please load your vector database first.")

        vector_db = Chroma(
            persist_directory=chroma_db_directory,
            embedding_function=self.get_embeddings(),
            collection_name=self.cfg.chroma.db_name,
            client_settings=self.chroma_settings,
        )

        return vector_db.as_retriever(
            search_kwargs={"k": TARGET_SOURCE_CHUNKS}
        )



    def initiate_document_injetion_pipeline(self):
        loaded_pdfs = self.load_pdfs()
        chunked_documents = self.split_documents(loaded_docs=loaded_pdfs)


        print("=> vector db initialised and created.")
        print("All done")

        client = chromadb.Client(self.chroma_settings)
        collection = client.get_collection(name="knowledge_base")
        print(f"Document count after restart: {collection.count()}")

