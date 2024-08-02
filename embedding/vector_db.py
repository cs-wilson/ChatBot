from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from .zhipuai_embedding import ZhipuAIEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
import os


class VectorDB:

    persist_directory = './vector_db/chroma/'

    def __init__(self, embedding_type: str):

        if embedding_type == 'openai':
            self.embedding = OpenAIEmbeddings()
        elif embedding_type == 'baidu':
            self.embedding = QianfanEmbeddingsEndpoint()
        elif embedding_type == 'zhipu':
            self.embedding = ZhipuAIEmbeddings()

    def build_vector_db(self, split_docs: list):

        if self.persist_directory is None:
            print("No persist directory specified, not persisting")
            os.makedirs(self.persist_directory, exist_ok=True)

        vectordb = Chroma.from_documents(
            # 为了速度，只选择前 20 个切分的 doc 进行生成；使用千帆时因QPS限制，建议选择前 5 个doc
            documents=split_docs,
            embedding=self.embedding,
            persist_directory=self.persist_directory  # 允许我们将persist_directory目录保存到磁盘上
        )
        print(f"向量库中存储的数量：{vectordb._collection.count()}")

    def get_vector_db(self):
        vectordb = Chroma(
            persist_directory=self.persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
            embedding_function=self.embedding
        )
        return vectordb
