import os
import json
import requests
from openai import OpenAI
from zhipuai import ZhipuAI


class Embedding:
    def __init__(self):
        pass

    def openai_embedding(self, text: str, api_key: str = None, model: str = None):

        # embedding model：'text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002'
        if model == None:
            model = "text-embedding-3-small"

        if api_key == None:
            # 获取环境变量 OPENAI_API_KEY
            api_key = os.environ["OPENAI_API_KEY"]

        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(input=text, model=model)
        return response.data[0].embedding

    def wenxin_embedding(self, text: str, api_key: str = None, secret_key: str = None):
        if api_key == None or secret_key == None:
            # 获取环境变量 wenxin_api_key、wenxin_secret_key
            api_key = os.environ["QIANFAN_AK"]
            secret_key = os.environ["QIANFAN_SK"]

        # 使用API Key、Secret Key向https://aip.baidubce.com/oauth/2.0/token 获取Access token
        url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={0}&client_secret={1}".format(
            api_key, secret_key
        )
        payload = json.dumps("")
        headers = {"Content-Type": "application/json",
                   "Accept": "application/json"}
        response = requests.request("POST", url, headers=headers, data=payload)

        # 通过获取的Access token 来embedding text
        url = (
            "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/embedding-v1?access_token="
            + str(response.json().get("access_token"))
        )
        input = []
        input.append(text)
        payload = json.dumps({"input": input})
        headers = {"Content-Type": "application/json"}

        response = requests.request("POST", url, headers=headers, data=payload)

        return json.loads(response.text)

    def zhipu_embedding(self, text: str, api_key: str = None):
        if api_key == None:
            # 获取环境变量 ZHIPUAI_API_KEY
            api_key = os.environ["ZHIPUAI_API_KEY"]
        client = ZhipuAI(api_key=api_key)
        response = client.embeddings.create(
            model="embedding-2",
            input=text,
        )
        return response.data[0].embedding
