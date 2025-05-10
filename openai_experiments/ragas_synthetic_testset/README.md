# [RAG Testset Generation](https://docs.ragas.io/en/stable/getstarted/rag_testset_generation/)

At the core there are 2 main operations that are performed to generate a testset.

1. KnowledgeGraph Creation:We first create a KnowledgeGraph using the documents you provide and use various Transformations to enrich the
knowledge graph with additional information that we can use to generate the testset. You can learn more about this from the core concepts section.

2. Testset Generation: We use the KnowledgeGraph to generate a set of scenarios. 
These scenarios are used to generate the testset. You can learn more about this from the core concepts section.
