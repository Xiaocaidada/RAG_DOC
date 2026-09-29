import os

from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.zhipuai import ZhipuAI

# 允许NLTK在代理下下载资源
os.environ["NLTK_ALLOW_PROXIED_URLOPEN"] = "1"
import nltk
# 预先下载依赖包
nltk.download('stopwords')
nltk.download('punkt_tab')
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.llm = OpenAILike(
    model="glm-5.3-flash",
    api_key=os.getenv("ZAI_API_KEY"),
    api_base="https://open.bigmodel.cn/api/paas/v4",
    temperature=0.1,
    timeout=120,
    is_chat_model=True,   # 标记是对话模型
    context_window=128000 # 手动指定GLM上下文，绕过自动识别
)

Settings.embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

file_path="../docs/C3/pdf/IPCC_AR6_WGII_Chapter03.pdf"
documents=SimpleDirectoryReader(input_files=[file_path]).load_data()

node_parser=SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

#句子窗口索引
sentence_nodes=node_parser.get_nodes_from_documents(documents)
sentence_index=VectorStoreIndex(sentence_nodes)

#常规分块索引
base_parser=SentenceSplitter(chunk_size=512)
base_nodes=base_parser.get_nodes_from_documents(documents)
base_index=VectorStoreIndex(base_nodes)

#构建查询引擎
sentence_query_engine=sentence_index.as_query_engine(
    similarity_top_k=2,
    node_postprocessors=[MetadataReplacementPostProcessor(target_metadata_key="window")],
)
base_query_engine=base_index.as_query_engine()

query="What are the concerns surrounding the AMOC?"
print("--- 句子窗口线索结果 ---")
window_response=sentence_query_engine.query(query)
print(f"回答：{window_response}")
print("--- 常规检索结果 ---")
base_response=base_query_engine.query(query)
print(f"回答：{base_response}")