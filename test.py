from config.config import Config
from models.generator import GeneratorModel
from embedding.embedding import Embedding
from data.loader import Loader
from utils.utils import Utils
import os


def main():
    user_input = "请你自我介绍一下自己！"

    # 加载配置文件
    config = Config()
    openai_api_key = config.parse_llm_api_key("openai")
    print("openai_api_key: ", openai_api_key)
    wenxin_api_key, wenxin_secret_key = config.parse_llm_api_key("wenxin")
    print("wenxin_api_key: ", wenxin_api_key)
    print("wenxin_secret_key: ", wenxin_secret_key)
    spark_app_id, spark_api_key, spark_secret_key = config.parse_llm_api_key(
        "spark")
    print("spark_api_key: ", spark_api_key)
    print("spark_app_id: ", spark_app_id)
    print("spark_secret_key: ", spark_secret_key)
    zhipu_api_key = config.parse_llm_api_key("zhipuai")
    print("zhipu_api_key: ", zhipu_api_key)

    # 加载生成器并获取生成结果
    openai_generator = GeneratorModel("openai", openai_api_key, "gpt-4o", 0.3)
    print("openai_generator: ", openai_generator)

    wenxin_generator = GeneratorModel(
        "wenxin", wenxin_api_key, "Yi-34B-Chat", 0.3, wenxin_secret_key
    )
    #  response = openai_generator.get_completion_openAI(user_input)
    #  print("response: ", response)

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

    knowledge_path = "./docs/"
    loader = Loader(knowledge_path)
    pdf_pages = loader.load_pdf("pumkin_book/pumpkin_book.pdf")
    print(
        f"载入后的变量类型为：{type(pdf_pages)}，",
        f"该 PDF 一共包含 {len(pdf_pages)} 页",
    )

    md_pages = loader.load_md("prompt_engineering/1. 简介 Introduction.md")
    print(
        f"载入后的变量类型为：{type(md_pages)}，",
        f"该 Markdown 一共包含 {len(md_pages)} 页",
    )

    # split_docs = loader.split_text(pdf_pages)
    # print(f"切分后的文件数量：{len(split_docs)}")
    # print("切分后的文件内容：", split_docs)

    for i in range(len(pdf_pages)):
        #   print(f"第 {i+1} 页内容：", pdf_pages[i].page_content)
        pdf_pages[i].page_content = loader.data_clean_del_slashN(
            pdf_pages[i].page_content
        )
    split_docs = loader.split_text(pdf_pages)
    print(f"切分后的文件数量：{len(split_docs)}")

    print(
        f"切分后的字符数（可以用来大致评估 token 数）：{sum([len(doc.page_content) for doc in split_docs])}")

    # 获取folder_path下所有文件路径，储存在file_paths里
    folder_path = "./docs/"
    utils = Utils()
    file_paths = utils.get_data_path(folder_path)
    print("file_paths: ", file_paths)

    texts = loader.file_loader(file_paths)
    text = texts[1]
    print(f"每一个元素的类型：{type(text)}.",
          f"该文档的描述性数据：{text.metadata}",
          f"查看该文档的内容:\n{text.page_content[0:]}",
          sep="\n------\n")

    split_docs = loader.split_text(texts)
    print(f"分割后的文档个数：{len(split_docs)}")

    vector_db = VectorDB("openai")
    # vector_db.build_vector_db(split_docs)

    vector_db_content = vector_db.get_vector_db()
    print(f"向量库中存储的数量：{vector_db_content._collection.count()}")
    # print(f"类型", str(type(vector_db_content)))

    # question = "什么是prompt engineering?"
    # docs = vector_db_content.similarity_search(question, k=3)
    # print(f"检索到的内容数：{len(docs)}")

    # for i, doc in enumerate(docs):
    #     print(f"检索到的第{i+1}个内容: \n {doc.page_content}",
    #           end="\n-----------------------------------------------------\n")

    retrieval = Retrieval()
    question = "什么是南瓜书？"
    response = retrieval.get_answer(
        question, vector_db_content)
    print("response: ", response)


if __name__ == "__main__":
    main()
