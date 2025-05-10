import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "helpers/data")


class OpenAiRAG:
  def __init__(self, model="gpt-4o-mini"):
    self.llm = ChatOpenAI(model=model)
    self.embeddings = OpenAIEmbeddings()
    self.doc_embeddings = None
    self.docs = None