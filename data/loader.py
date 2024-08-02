from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import re


class Loader:
    def __init__(self, base_path):
        self.path = base_path

    def load_pdf(self, pdf_path: str):

        path = os.path.join(self.path, pdf_path)
        pdf_loader = PyMuPDFLoader(path)
        pdf_pages = pdf_loader.load()
        return pdf_pages

    def load_md(self, md_path: str):

        path = os.path.join(self.path, md_path)
        md_loader = UnstructuredMarkdownLoader(path)
        md_content = md_loader.load()
        return md_content

    def data_clean_del_slashN(self, content):
        pattern = re.compile(
            r"[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]", re.DOTALL)
        cleaned_content = re.sub(
            pattern, lambda match: match.group(0).replace("\n", ""), content
        )
        return cleaned_content

    def data_clean_replace_slashN(self, content):
        cleaned_content = content.replace("\n\n", "\n")
        return cleaned_content

    def split_text(self, content, CHUNK_SIZE=500, OVERLAP_SIZE=50):
        # 使用递归字符文本分割器
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP_SIZE
        )
        split_docs = text_splitter.split_documents(content)
        return split_docs

    def file_loader(self, file_paths: list):

        # 遍历文件路径并把实例化的loader存放在loaders里
        loaders = []
        for file_path in file_paths:
            file_type = file_path.split('.')[-1]
            if file_type == 'pdf':
                loaders.append(PyMuPDFLoader(file_path))
            elif file_type == 'md':
                loaders.append(UnstructuredMarkdownLoader(file_path))
        texts = []
        for loader in loaders:
            texts.extend(loader.load())
        return texts
