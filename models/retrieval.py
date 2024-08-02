from langchain.chains import RetrievalQA
# from models.generator import GeneratorModel
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


class Retrieval:
    def init(self):
        pass

    def get_answer(self, question: str, vectordb: object):
        # 如果上下文里没有答案，你可以自己思考一下，给出答案，但不要提供错误的信息源。

        template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。
        如果上下文里没有答案，你可以自己思考一下，给出答案，但不要提供错误的信息源。
        最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
        {context}
        问题: {question}
        """

        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],
                                         template=template)

        llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

        qa_chain = RetrievalQA.from_chain_type(llm,
                                               retriever=vectordb.as_retriever(),
                                               return_source_documents=True,
                                               chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

        # response = qa_chain({"query": question})

        response = qa_chain.invoke({"query": question})
        return response

    def get_answer_from_history(self, question: str, vectordb: object):
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        retriever = vectordb.as_retriever()
        qa = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=retriever,
            memory=memory
        )
        # result = qa({"question": question})
        result = qa.invoke({"question": question})
        return result['answer']
