import os
import pandas as pd
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.schema import IndexNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine, BaseQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.response import Response
# 新增导入回调管理器
from llama_index.core.callbacks import CallbackManager

load_dotenv()

Settings.llm = OpenAILike(
    model="glm-5.3-flash",
    api_key=os.getenv("ZAI_API_KEY"),
    api_base="https://open.bigmodel.cn/api/paas/v4",
    temperature=0.1,
    timeout=120,
    is_chat_model=True,
    context_window=128000
)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")


# 自定义简易Pandas查询引擎
class SimplePandasQueryEngine(BaseQueryEngine):
    def __init__(self, df: pd.DataFrame, llm, verbose: bool = True):
        self.df = df
        self.llm = llm
        self.verbose = verbose
        # 提示词：让LLM生成安全的pandas查询代码
        self.prompt_template = PromptTemplate("""
你是pandas数据分析专家。
给定DataFrame的列名：{df_columns}
用户问题：{query_str}

只输出一行python pandas代码，不要解释，不要markdown。
只能使用df变量，不能修改原始数据，不能import包。
如果无法回答，直接返回字符串"NO_RESULT"
""")
        # 修复：传入callback_manager
        super().__init__(callback_manager=CallbackManager())

    def _query(self, query_bundle):
        query_str = query_bundle.query_str
        df_cols = list(self.df.columns)

        # 调用LLM生成pandas代码
        prompt = self.prompt_template.format(df_columns=df_cols, query_str=query_str)
        code_resp = self.llm.complete(prompt)
        code = code_resp.text.strip()
        if self.verbose:
            print(f"\n=====生成代码=====\n{code}\n==================")

        if code == "NO_RESULT":
            return Response(response="无法从表格中查询到对应数据")

        # 安全执行代码，限制局部命名空间
        local_vars = {"df": self.df.copy(), "pd": pd}
        try:
            result = eval(code, {}, local_vars)
            # 把结果转为字符串返回
            answer = str(result)
        except Exception as e:
            answer = f"执行查询代码失败，错误信息：{str(e)}"
        return Response(response=answer)

    async def _aquery(self, query_bundle):
        return self._query(query_bundle)

    def _get_prompt_modules(self):
        return {}


# ========== 下面逻辑完全不变 ==========
excel_file = "../docs/C3/excel/movie.xlsx"
xls = pd.ExcelFile(excel_file)

df_query_engines = {}
all_nodes = []

for sheet_name in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet_name)
    query_engine = SimplePandasQueryEngine(df=df, llm=Settings.llm, verbose=True)

    year = sheet_name.replace('年份_', '')
    summary = f"这个表格包含了年份为 {year} 的电影信息，可以用来回答关于这一年电影的具体问题。"
    node = IndexNode(text=summary, index_id=sheet_name)
    all_nodes.append(node)
    df_query_engines[sheet_name] = query_engine

vector_index = VectorStoreIndex(all_nodes)
vector_retriever = vector_index.as_retriever(similarity_top_k=1)

recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever},
    query_engine_dict=df_query_engines,
    verbose=True
)
query_engine = RetrieverQueryEngine.from_args(recursive_retriever)

query = "1988年评分最少的电影是哪一部"
print(f"查询: {query}")
response = query_engine.query(query)
print(f"回答：{response}")
