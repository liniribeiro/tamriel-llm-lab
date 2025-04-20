import os

from omegaconf import DictConfig

from end_to_end_llm.src.knowledgebase import MyKnowledgeBase
from end_to_end_llm.src.my_gpt4all import MyGPT4ALL

# import all the langchain modules
from langchain.chains import RetrievalQA


def retrieval_qa(query: str, cfg: DictConfig):
    """
    Retrieves a query from the persistent vector database and initializes a QA chain.

    Args:
        query (str): The query to retrieve from the vector database.

    Returns:
        dict: The query result and the QA chain.
    """
    # Initialize the LLM
    llm = MyGPT4ALL(
        model_folder_path=cfg.my_model.folder_path,
        model_name=cfg.my_model.model_name,
        allow_download=cfg.my_model.allow_download,
    )

    # Initialize the knowledge base
    kb = MyKnowledgeBase(cfg=cfg)
    retriever = kb.return_retriever_from_persistant_vector_db()

    # Initialize the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )

    return qa_chain.invoke(query)