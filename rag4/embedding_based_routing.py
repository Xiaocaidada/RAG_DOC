import os

import numpy as np
from langchain_community.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

sichuan_route_prompt="你是一位处理川菜的专家。用户的问题是关于麻辣、辛香、重口味的菜肴，例如水煮鱼、麻婆豆腐、鱼香肉丝、宫保鸡丁、花椒、海椒等。"
cantonese_route_prompt="你是一位处理粤菜的专家。用户的问题是关于清淡、鲜美、原汁原味的菜肴，例如白切鸡、老火靓汤、虾饺、云吞面等。"


route_prompts=[sichuan_route_prompt,cantonese_route_prompt]
route_names=["川菜","粤菜"]



embeddings=HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
route_prompt_embeddings=embeddings.embed_documents(route_prompts)
print(f"已定义 {len(route_names)}个路由: {','.join(route_names)}")


llm=ChatOpenAI(
    model="glm-5.3-flash",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    api_key=os.getenv("ZAI_API_KEY"),
    temperature=0.6,
)


sichuan_chain=(
    PromptTemplate.from_template("你是一位川菜大厨。请用正宗的川菜做法，回答关于「{query}」的问题。")
    | llm
    | StrOutputParser()
)
cantonese_chain=(
    PromptTemplate.from_template("你是一位粤菜大厨。请用经典的粤菜做法，回答关于「{query}」的问题。")
    | llm
    | StrOutputParser()
)

route_map={'川菜': sichuan_chain,"粤菜": cantonese_chain}
print("川菜和粤菜的处理链创建成功。\n")


def route(info):
    query_embedding=embeddings.embed_query(info['query'])
    similarity_scores=cosine_similarity([query_embedding],route_prompt_embeddings)
    chosen_route_index=np.argmax(similarity_scores)
    chosen_route_name=route_names[chosen_route_index]
    print(f"路由决策: 检测到问题与 {chosen_route_name}最相似")
    chosen_chain=route_map[chosen_route_name]
    return chosen_chain.invoke(info)


full_chain=RunnableLambda(route)

# 4. 运行演示查询
demo_queries = [
    "水煮鱼怎么做才嫩？",        # 应该路由到川菜
    "如何做一碗清淡的云吞面？",    # 应该路由到粤菜
    "麻婆豆腐的核心调料是什么？",  # 应该路由到川菜
]

for i, query in enumerate(demo_queries, 1):
    print(f"\n--- 问题 {i}: {query} ---")
    try:
        # 传入字典，full_chain 会直接返回最终答案
        result = full_chain.invoke({"query": query})
        print(f"回答: {result}")
    except Exception as e:
        print(f"执行错误: {e}")

