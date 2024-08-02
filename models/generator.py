"""
Dec: 定义生成器模型，主要是利用大模型API来实现文本生成
Name: generator.py
"""

from typing import Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import QianfanLLMEndpoint
from langchain_community.llms import SparkLLM
from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage
# from langchain_community.llms import Runnable


class GeneratorModel:

    platform: str
    api_key: str
    model_name: str
    temperature: float
    appid: Optional[str]
    secret_key: Optional[str]
    proxy: Optional[str]
    client: Optional[Any]

    def __init__(
        self,
        platform: str,
        api_key: str,
        model_name: str,
        temperature: float,
        appid: str = None,
        secret_key: str = None,
        proxy: str = None,
    ):
        self.platform = platform
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key
        self.appid = appid
        self.secret_key = secret_key
        self.proxy = proxy
        self.client = None
        self.init_client()

    def init_client(self):
        if self.platform == "openai":
            self.client = ChatOpenAI(
                api_key=self.api_key,
                model=self.model_name,
                temperature=self.temperature,
            )
        elif self.platform == "wenxin":
            self.client = QianfanLLMEndpoint(
                qianfan_ak=self.api_key,
                qianfan_sk=self.secret_key,
                model=self.model_name,
                temperature=self.temperature,
            )
        elif self.platform == "spark":
            SPARKAI_URL = "wss://spark-api.xf-yun.com/v3.5/chat"

            # spark_api_url = self.gen_spark_params(model="v3.5")["spark_url"]
            self.client = ChatSparkLLM(
                spark_api_url=SPARKAI_URL,
                spark_app_id=self.appid,
                spark_api_key=self.api_key,
                spark_api_secret=self.secret_key,
                spark_temperature=self.temperature,
            )
        elif self.platform == "zhipuai":
            self.client = ZhipuaiApi(self.api_key)
        else:
            raise ValueError(f"platform{self.platform} not support!!!")

    def generate(self, user_input: str):
        if self.platform == "openai":
            return self.get_completion_openAI(user_input)

    def get_completion_openAI(self, user_input: str):
        chain = self.build_prompt | self.client | self.parse_output
        result = chain.invoke(user_input)
        return result

    def build_prompt(self, user_input: str, name: str = "文兴"):
        # template = "你是一个翻译助手，可以帮助我将 {input_language} 翻译成 {output_language}."
        template = "你是一个智能助手，你的名字叫 {name}."

        chat_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", template),
                ("human", "{user_input}"),
            ]
        )
        # messages  = chat_prompt.format_messages(input_language="中文", output_language="英文", user_input=user_input, name=name)
        messages = chat_prompt.format_messages(
            user_input=user_input, name=name)

        return messages

    def parse_output(self, output: str):
        output_parser = StrOutputParser()
        result = output_parser.invoke(output)
        return result

    def gen_spark_params(self, model):
        """
        构造星火模型请求参数
        """

        spark_url_tpl = "wss://spark-api.xf-yun.com/{}/chat"
        model_params_dict = {
            # v1.5 版本
            "v1.5": {
                "domain": "general",  # 用于配置大模型版本
                "spark_url": spark_url_tpl.format("v1.1"),  # 云端环境的服务地址
            },
            # v2.0 版本
            "v2.0": {
                "domain": "generalv2",  # 用于配置大模型版本
                "spark_url": spark_url_tpl.format("v2.1"),  # 云端环境的服务地址
            },
            # v3.0 版本
            "v3.0": {
                "domain": "generalv3",  # 用于配置大模型版本
                "spark_url": spark_url_tpl.format("v3.1"),  # 云端环境的服务地址
            },
            # v3.5 版本
            "v3.5": {
                "domain": "generalv3.5",  # 用于配置大模型版本
                "spark_url": spark_url_tpl.format("v3.5"),  # 云端环境的服务地址
            },
        }
        return model_params_dict[model]
