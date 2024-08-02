from config.config import Config
from models.generator import GeneratorModel
from embedding.embedding import Embedding
from data.loader import Loader
from utils.utils import Utils
from embedding.vector_db import VectorDB
from models.retrieval import Retrieval
import os


def main():
    user_input = "请你自我介绍一下自己！"

    # 加载配置文件
    config = Config()
    openai_api_key = config.parse_llm_api_key("openai")
    # print("openai_api_key: ", openai_api_key)
    wenxin_api_key, wenxin_secret_key = config.parse_llm_api_key("wenxin")
    # print("wenxin_api_key: ", wenxin_api_key)
    # print("wenxin_secret_key: ", wenxin_secret_key)
    spark_app_id, spark_api_key, spark_secret_key = config.parse_llm_api_key(
        "spark")
    # print("spark_api_key: ", spark_api_key)
    # print("spark_app_id: ", spark_app_id)
    # print("spark_secret_key: ", spark_secret_key)
    zhipu_api_key = config.parse_llm_api_key("zhipuai")
    # print("zhipu_api_key: ", zhipu_api_key)

    # 加载生成器并获取生成结果
    openai_generator = GeneratorModel("openai", openai_api_key, "gpt-4o", 0.3)
    # print("openai_generator: ", openai_generator)
    user_input = "阿兴是谁？"
    response = openai_generator.get_completion_openAI(user_input)
    print("response From AI basic: ", response)

    wenxin_generator = GeneratorModel(
        "wenxin", wenxin_api_key, "Yi-34B-Chat", 0.3, wenxin_secret_key
    )

    #  spark_generator = GeneratorModel(
    #      "spark", spark_app_id, spark_api_key, spark_secret_key, 0.3
    #  )
    #  print("spark_generator: ", spark_generator)
    # 加载embedding模型
    embedding = Embedding()
    response = embedding.openai_embedding(
        user_input, openai_api_key, "text-embedding-3-small"
    )
    #  print("response: ", response)

    # 百度不可用，api key不可用
    #  baidu_embedding = embedding.wenxin_embedding(
    #      user_input, wenxin_api_key, wenxin_secret_key
    #  )
    #  print("baidu_embedding: ", baidu_embedding)

    zhipu_embedding = embedding.zhipu_embedding(user_input, zhipu_api_key)
    #  print("zhipu_embedding: ", zhipu_embedding)

    loader = Loader("./docs/")

    # 获取folder_path下所有文件路径，储存在file_paths里
    folder_path = "./docs/"
    utils = Utils()
    file_paths = utils.get_data_path(folder_path)
    # print("file_paths: ", file_paths)

    texts = loader.file_loader(file_paths)
    # text = texts[1]
    # print(f"每一个元素的类型：{type(text)}.",
    #       f"该文档的描述性数据：{text.metadata}",
    #       f"查看该文档的内容:\n{text.page_content[0:]}",
    #       sep="\n------\n")

    split_docs = loader.split_text(texts)
    # print(f"分割后的文档个数：{len(split_docs)}")

    vector_db = VectorDB("openai")
    # vector_db.build_vector_db(split_docs)

    vector_db_content = vector_db.get_vector_db()
    # print(f"向量库中存储的数量：{vector_db_content._collection.count()}")
    # print(f"类型", str(type(vector_db_content)))

    # question = "什么是prompt engineering?"
    # docs = vector_db_content.similarity_search(question, k=3)
    # print(f"检索到的内容数：{len(docs)}")

    retrieval = Retrieval()
    question = "阿兴是谁？"
    question_back = "什么是南瓜书？"
    # docs = vector_db_content.similarity_search(question_2, k=3)
    # print(f"检索到的内容数：{docs}")

    response = retrieval.get_answer(question,
                                    vector_db_content
                                    )
    print("response from DB: ", response["result"])
    # question1 = "什么是prompt engineering?"
    # response1 = retrieval.get_answer_from_history(question1,
    #                                               vector_db_content
    #                                               )
    # print("response1: ", response1)
    question2 = "为什么这门课需要教这方面的知识？"
    response2 = retrieval.get_answer_from_history(question2,
                                                  vector_db_content
                                                  )
    print("response2: ", response2)


if __name__ == "__main__":
    main()
