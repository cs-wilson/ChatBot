# Desc: 用于加载项目的相关配置信息，包括数据库配置、API Key等
# Name: config.py
# Function: 用于加载项目的环境和配置信息，包括数据库配置、API Key等

import os
from dotenv import load_dotenv, find_dotenv
from langchain.utils import get_from_dict_or_env


class Config:

    def __init__(self):
        pass

    # API Key
    def parse_llm_api_key(self, model: str, env_file: dict = None):
        """
        通过 model 和 env_file 的来解析平台参数
        """
        if env_file == None:
            # 自动查找 .env 文件并加载环境变量
            env_path = find_dotenv()
            if env_path:
                # print(f"Found .env file at {env_path}")
                _ = load_dotenv(env_path)
            else:
                print("No .env file found")
        env_file = os.environ
        # print(env_file["QIANFAN_AK"])
        if model == "openai":
            return env_file["OPENAI_API_KEY"]
        elif model == "wenxin":
            return env_file["QIANFAN_AK"], env_file["QIANFAN_SK"]
        elif model == "spark":
            return (
                env_file["SPARK_APPID"],
                env_file["SPARK_API_KEY"],
                env_file["SPARK_API_SECRET"],
            )
        elif model == "zhipuai":
            # return get_from_dict_or_env(env_file, "zhipuai_api_key", "ZHIPUAI_API_KEY")
            return env_file["ZHIPUAI_API_KEY"]
        else:
            raise ValueError(f"model{model} not support!!!")
