
from dotenv import load_dotenv
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas.testset import TestsetGenerator

from langchain_community.document_loaders import JSONLoader

from openai_experiments.base import DATA_DIR


if __name__ == "__main__":
    load_dotenv()


    def metadata_func(record: dict, metadata: dict) -> dict:
        """
        Update the metadata of the documents to include the source and sequence number.
        """
        metadata.update(
            {
                "headlines": [record.get("title", "Untitled")]
            }
        )
        return metadata



    # Load books
    loader = JSONLoader(
        file_path=f"{DATA_DIR}/books.json",
        jq_schema=".[]",  # JQ filter to extract the 'text' field
        text_content=False,
        content_key="content",
        metadata_func=metadata_func,
    )
    documents = loader.load()

    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    # Generate Sample
    generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
    dataset = generator.generate_with_langchain_docs(documents, testset_size=10)
    df = dataset.to_pandas()

    # Save to CSV
    df.to_csv("testset.csv")

