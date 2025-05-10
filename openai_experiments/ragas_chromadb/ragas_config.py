from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from datasets import Dataset

from openai_experiments.base import OpenAiRAG


# Define prompt template
template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use two sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

def get_dataset(rag: OpenAiRAG, retriever):
    # Setup RAG pipeline
    rag_chain = (
        {"context": retriever,  "question": RunnablePassthrough()}
        | prompt
        | rag.llm
        | StrOutputParser()
    )


    questions = ["Who is Alduin?",
                 "Who is High Priest?",
                 "Who is Alini",
                ]
    ground_truths = ["Alduin is a Dragon",
                    "Alexandre Simon is the High Priest Author of the book",
                    "Alini is unknown"]
    answers = []
    contexts = []

    # Inference
    for query in questions:
      answers.append(rag_chain.invoke(query))
      contexts.append([docs.page_content for docs in retriever.invoke(query)])

    # To dict
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "reference": ground_truths
    }

    # Convert dict to dataset
    dataset = Dataset.from_dict(data)

    return dataset

def ragas_evaluation(rag: OpenAiRAG, retriever):

    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )

    dataset = get_dataset(rag, retriever)
    result = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
    )

    df = result.to_pandas()
    print(df.values)

    df.to_csv("data/ragas_output.csv", index=False)