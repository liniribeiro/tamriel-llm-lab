from dotenv import load_dotenv

from openai_experiments.base import OpenAiRAG
from openai_experiments.ragas_chromadb.download_data import download_data
from openai_experiments.ragas_chromadb.retriever import chunk_and_embed_data, get_vectorstore
from openai_experiments.ragas_chromadb.ragas_config import ragas_evaluation

if __name__ == "__main__":
    rag = OpenAiRAG()
    # download_data()
    # print("Data downloaded successfully.")
    # file_name = "data/The Alduin_Akatosh Dichotomy.txt"
    # vectorstore = chunk_and_embed_data(rag, file_name)
    # print("Data chunked and embedded successfully.")

    retriever = get_vectorstore(rag).as_retriever()
    ragas_evaluation(rag, retriever)

