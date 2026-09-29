from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

texts=[
    "张三是法外狂徒",
    "FAISS是一个用于高效相似性搜索和密集向量聚类的库。",
    "LangChain是一个用于开发由语言模型驱动的应用程序的框架。"
]
docs=[Document(page_content=text) for text in texts]
embeddings=HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

vectorStore=FAISS.from_documents(docs,embeddings)

local_faiss_path="./faiss_index_store"
vectorStore.save_local(local_faiss_path)
print(f"FAISS index has been saved to {local_faiss_path}")


#加载本地索引执行查询操作
loaded_vector_store=FAISS.load_local(local_faiss_path,embeddings,allow_dangerous_deserialization=True)

query=("FAISS是做什么的")
result=loaded_vector_store.similarity_search(query,k=1)
print(f"\n查询: {result}")
print("相似度最高文档：")
for doc in result:
    print(f"- {doc.page_content}")
