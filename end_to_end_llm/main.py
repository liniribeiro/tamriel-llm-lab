import hydra
from omegaconf import DictConfig

from end_to_end_llm.src.my_gpt4all import MyGPT4ALL
from end_to_end_llm.src.retriever import retrieval_qa


@hydra.main(config_path='./configs/model', config_name='default', version_base='1.1')
def main_mygpt4all(cfg):
    """
    Main function to run the MyGPT4ALL model.
    """
    chat_model = MyGPT4ALL(
        model_folder_path=hydra.utils.to_absolute_path(cfg.gpt4all.gpt4all_model_folder_path),
        model_name=cfg.gpt4all.gpt4all_model_name,
        allow_download=False,

    )

    while True:
        query = input('Enter your Query: ')
        if query == 'exit':
            break
        # use hydra to fill the **kwargs
        response = chat_model(
            query,
        )
        print(response)

@hydra.main(config_path='./configs/knowledge_database', config_name='default', version_base=None)
def main_knowledgebase(cfg: DictConfig):
    """
    Main function to run the knowledge base retrieval and question answering.
    """
    while True:
        query = input("What's on your mind: ")
        if query == 'exit':
            break
        result = retrieval_qa(query, cfg)
        answer, docs = result['result'], result['source_documents']

        print(answer)

        print("#" * 30, "Sources", "#" * 30)
        for document in docs:
            print("\n> SOURCE: " + document.metadata["source"] + ":")
            # print(document.page_content)
        print("#" * 30, "Sources", "#" * 30)


if __name__ == '__main__':
    # main_mygpt4all()
    main_knowledgebase()



