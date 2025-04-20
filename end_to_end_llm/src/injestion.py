""""
File that generates the knowledge base embeddings
"""
import hydra
from omegaconf import DictConfig

from knowledgebase import MyKnowledgeBase



@hydra.main(config_path='../configs/knowledge_database', config_name='default', version_base=None)
def inject_data(cfg: DictConfig):
    # kb is here knowledge base
    kb = MyKnowledgeBase(cfg)

    kb.initiate_document_injetion_pipeline()

inject_data()