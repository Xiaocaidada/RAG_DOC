import json
import logging
import os

from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers import SelfQueryRetriever
from langchain_community.document_loaders import BiliBiliLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from llama_index.llms.openai_like import OpenAILike
from openai import OpenAI

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

# 4. 初始化LLM客户端
client = OpenAI(
    base_url="https://open.bigmodel.cn/api/paas/v4",
    api_key=os.getenv("ZAI_API_KEY")
)

queries=[
    "时间最短的视频","播放量最高的视频"
]
all_documents=vectorstore.similarity_search("",k=len(bili))

for query in queries:
    print(f"\n查询: {query}")
    prompt=f"""你是一个智能助手，请将用户的问题转换成一个用于排序视频的JSON指令。

你需要识别用户想要排序的字段和排序方向。
- 排序字段必须是 'view_count' (观看次数) 或 'length' (时长) 之一。
- 排序方向必须是 'asc' (升序) 或 'desc' (降序) 之一。

例如:
- '时间最短的视频' 或 '哪个视频时间最短' 应转换为 {{"sort_by": "length", "order": "asc"}}
- '播放量最高的视频' 或 '哪个视频最火' 应转换为 {{"sort_by": "view_count", "order": "desc"}}

请根据以下问题生成JSON指令:
原始问题: "{query}"

JSON指令:"""

    response=client.chat.completions.create(
        model="glm-5.3-flash",
        messages=[{"role":"user","content":prompt}]
    )
    try:
        instruction_str=response.choices[0].message.content
        instruction=json.loads(instruction_str)
        print(f"---- 生成的排序指令: {instruction} ---")
        sorted_by=instruction.get("sort_by")
        order=instruction.get("order")
        if  sorted_by in ['length','view_count'] and order in ['asc','desc']:
            reverse_order=(order=="desc")
            sorted_docs=sorted(all_documents,key=lambda doc: doc.metadata.get(sorted_by,0),reverse=reverse_order)

            if sorted_docs:
                doc=sorted_docs[0]
                title = doc.metadata.get('title', '未知标题')
                author = doc.metadata.get('author', '未知作者')
                view_count = doc.metadata.get('view_count', '未知')
                length = doc.metadata.get('length', '未知')
                print(f"标题: {title}")
                print(f"作者: {author}")
                print(f"观看次数: {view_count}")
                print(f"时长: {length}秒")
                print("=" * 50)
            else:
                print("没有找到任何数据")
        else:
            print("生成指令无效,无法执行排序")
    except (json.JSONDecodeError,KeyError) as e:
        print(f"解析或者执行指令错误: {e}")












