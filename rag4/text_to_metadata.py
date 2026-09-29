import logging
import os

from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers import SelfQueryRetriever
from langchain_community.document_loaders import BiliBiliLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from llama_index.llms.openai_like import OpenAILike

logging.basicConfig(level=logging.INFO)



# 视频url
video_urls = [
    "https://www.bilibili.com/video/BV1Bo4y1A7FU",
    "https://www.bilibili.com/video/BV1ug4y157xA",
    "https://www.bilibili.com/video/BV1yh411V7ge",
]

bili=[]
try:
    loader=BiliBiliLoader(video_urls=video_urls)
    docs=loader.load()

    #重写每个doc的metadata
    for doc in docs:
        original=doc.metadata
        metadata = {
            'title': original.get('title', '未知标题'),
            'author': original.get('owner', {}).get('name', '未知作者'),
            'source': original.get('bvid', '未知ID'),
            'view_count': original.get('stat', {}).get('view', 0),
            'length': original.get('duration', 0),
        }
        doc.metadata=metadata
        bili.append(doc)

except Exception as e:
    print(f"加载BiliBili视频失败: {str(e)}")

if not bili:
    print("没有成功加载任何视频，程序退出")
    exit(0)


embed_model=HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore=Chroma.from_documents(bili,embed_model)

metadata_field_info=[
    AttributeInfo(
        name="title",
        description="视频标题(字符串)",
        type="string",
    ),
    AttributeInfo(
        name="author",
        description="视频作者(字符串)",
        type="string"
    ),
    AttributeInfo(
        name="view_count",
        description="视频观看次数(整数)",
        type="integer"
    ),
    AttributeInfo(
        name="length",
        description="视频长度",
        type="integer"
    )
]

llm = ChatOpenAI(
    model="glm-5.3-flash",
    api_key=os.getenv("ZAI_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4",
    temperature=0.1,
    timeout=120,
)

retriever=SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="记录视频标题、作者、观看次数等信息的视频元数据",
    metadata_field_info=metadata_field_info,
    enable_limit=True,
    verbose=True
)

queries=[
    "时间最短的视频","市场大于600秒的视频"
]

for query in queries:
    print(f"\n查询: {query}")
    results=retriever.invoke(query)
    if results:
        for doc in results:
            title = doc.metadata.get('title', '未知标题')
            author = doc.metadata.get('author', '未知作者')
            view_count = doc.metadata.get('view_count', '未知')
            length = doc.metadata.get('length', '未知')
            print(f"标题: {title}")
            print(f"作者: {author}")
            print(f"观看次数: {view_count}")
            print(f"时长: {length}秒")
            print("="*50)
    else:
        print("未匹配到视频")










