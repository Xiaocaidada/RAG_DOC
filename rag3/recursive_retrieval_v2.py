import os

import pandas as pd
from dotenv import load_dotenv
from llama_cloud import MetadataFilters
from llama_index.core import Settings, Document, VectorStoreIndex
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.zhipuai import ZhipuAI

load_dotenv()

Settings.llm = OpenAILike(
    model="glm-5.3-flash",
    api_key=os.getenv("ZAI_API_KEY"),
    api_base="https://open.bigmodel.cn/api/paas/v4",
    temperature=0.1,
    timeout=120,
    is_chat_model=True,   # 标记是对话模型
    context_window=128000 # 手动指定GLM上下文，绕过自动识别
)
Settings.embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")

excel_file="../docs/C3/excel/movie.xlsx"
xls=pd.ExcelFile(excel_file)

summary_docs=[]
content_docs=[]

print("开始加载和处理Excel文件...")
for sheet_name in xls.sheet_names:
    df=pd.read_excel(xls,sheet_name=sheet_name)

    #数据清洗
    if '评分人数' in df.columns:
        df['评分人数']=df['评分人数'].astype(str).str.replace('人评价','').str.strip()
        df['评分人数']=pd.to_numeric(df['评分人数'],errors='coerce').fillna(0).astype(int)
    year=sheet_name.replace('年份','')
    summary_text=f"这个表格包含了年份为 {year} 的电影信息，包括电影名称、导演、评分、评分人数等。"
    summary_doc=Document(text=summary_text,metadata={"sheet_name":sheet_name})
    summary_docs.append(summary_doc)

    content_text=df.to_string(index=False)
    content_doc=Document(text=content_text,metadata={"sheet_name":sheet_name})
    content_docs.append(content_doc)


#为摘要创建索引
summary_index=VectorStoreIndex(summary_docs)
#为内容创建索引
content_index=VectorStoreIndex(content_docs)


# 定义两部查询逻辑
def query_safe_recursive(query_str):
    print("--- 开始执行查询 ---")
    print(f"查询：{query_str}")


    #第一步查询
    print("\n第一步：在摘要索引中进行路由...")
    summary_retriever=VectorIndexRetriever(index=summary_index,similarity_top_k=1)
    retrieved_nodes=summary_retriever.retrieve(query_str)
    if not retrieved_nodes:
        return "抱歉，未能找到相关电影年份信息"

    matched_sheet_name=retrieved_nodes[0].node.metadata['sheet_name']
    print(f"路由结果：匹配到工作表 -> {matched_sheet_name}")
    # 第二步：检索 - 在内容索引中根据工作表名称过滤并检索具体内容
    print("\n第二步：在内容索引中检索具体信息...")
    content_retriever=VectorIndexRetriever(
        index=content_index,
        similarity_top_k=1,
        filters=MetadataFilters(filters=[ExactMatchFilter(key="sheet_name",value=matched_sheet_name)])
    )
    query_engine=RetrieverQueryEngine.from_args(content_retriever)
    response=query_engine.query(query_str)
    print("--- 查询执行结束 ---\n")
    return response


query="1994年评分人数最少的电影是哪一部？"
response=query_safe_recursive(query)
print(f"最终回答：{response}")