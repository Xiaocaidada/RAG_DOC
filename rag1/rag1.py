import os
# 允许NLTK走代理拉取资源
os.environ["NLTK_ALLOW_PROXIED_URLOPEN"] = "1"
import nltk
# 下载需要的数据包
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
file_path="../docs/C1/markdown/easy-rl-chapter1.md"
# 加载文档到内存
docs=UnstructuredMarkdownLoader(file_path).load()
print(f"文件加载到内存")
# 将文档进行切换
text_splitter=RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=100)
chunks=text_splitter.split_documents(docs)
print(f"文档被切割成块")


#中文嵌入模型
embeddings=HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
# 构建内存向量库
vectorStore=InMemoryVectorStore(embeddings)
vectorStore.add_documents(chunks)


# 提示词模板
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

上下文:
{context}

问题: {question}

回答:"""
)

llm=ChatOpenAI(
    model="glm-4.7-flash-free",
    temperature=0.7,
    max_tokens=4096,
    api_key=os.getenv("AIHUBMIX_KEY"),
    base_url=os.getenv("AIHUBMIX_URL")
)
question="文中举了哪些例子"

# 检索
retrieved_docs=vectorStore.similarity_search(question,k=3)
docs_content="\n\n".join(doc.page_content for doc in retrieved_docs)
print(f"检索到内容: {docs_content}")

answer=llm.invoke(prompt.format(question=question,context=docs_content))

# 模型输出结果
print(f">>> 模型输出：{answer.content}")


