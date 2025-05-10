import asyncio

from dotenv import load_dotenv
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.metrics import AspectCritic, LLMContextRecall, Faithfulness, FactualCorrectness

from openai_experiments.ragas_simple_rag.rag import RAG



def load_data(rag):
    sample_docs = [
        "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
        "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
        "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
        "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'.",
        "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine."
    ]

    # Load documents
    rag.load_documents(sample_docs)

    # Query and retrieve the most relevant document
    query = "Who introduced the theory of relativity?"
    relevant_doc = rag.get_most_relevant_docs(query)

    # Generate an answer
    answer = rag.generate_answer(query, relevant_doc)

    print(f"Query: {query}")
    print(f"Relevant Document: {relevant_doc}")
    print(f"Answer: {answer}")

def evaluate_metric(rag, evaluator_llm):
    sample_queries = [
        "Who introduced the theory of relativity?",
        "Who was the first computer programmer?",
        "What did Isaac Newton contribute to science?",
        "Who won two Nobel Prizes for research on radioactivity?",
        "What is the theory of evolution by natural selection?"
    ]

    expected_responses = [
        "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
        "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine.",
        "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
        "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
        "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'."
    ]

    dataset = []

    for query, reference in zip(sample_queries, expected_responses):
        relevant_docs = rag.get_most_relevant_docs(query)
        response = rag.generate_answer(query, relevant_docs)
        dataset.append(
            {
                "user_input": query,
                "retrieved_contexts": relevant_docs,
                "response": response,
                "reference": reference
            }
        )

    evaluation_dataset = EvaluationDataset.from_list(dataset)

    result = evaluate(dataset=evaluation_dataset,metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],llm=evaluator_llm)

    df = result.to_pandas()
    df.to_csv("ragas_output.csv", index=False)

if __name__ == "__main__":
    load_dotenv()
    # Initialize RAG instance
    rag = RAG()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    evaluator_llm = LangchainLLMWrapper(llm)

    load_data(rag)

    evaluate_metric(rag, evaluator_llm)
