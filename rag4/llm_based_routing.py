import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch
from langchain_openai import ChatOpenAI



llm=ChatOpenAI(
    model="glm-5.3-flash",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    api_key=os.getenv("ZAI_API_KEY"),
    temperature=0.6,
)


sichuan_prompt=ChatPromptTemplate.from_template(
    "你是一位川菜大厨。请用正宗的川菜做法，回答关于「{question}」的问题。"
)
sichuan_chain=sichuan_prompt | llm | StrOutputParser()


cantonese_prompt=ChatPromptTemplate.from_template(
    "你是一位粤菜大厨。请用经典的粤菜做法，回答关于「{question}」的问题。"
)
cantonese_chain=cantonese_prompt | llm | StrOutputParser()

general_prompt=ChatPromptTemplate.from_template(
    "你是一个美食助手。请回答关于「{question}」的问题。"
)
general_chain=general_prompt | llm | StrOutputParser()


# 路由链
classifier_prompt=ChatPromptTemplate.from_template(
    """根据用户问题中提到的菜品，将其分类为：['川菜', '粤菜', 或 '其他']。
    不要解释你的理由，只返回一个单词的分类结果。
    问题: {question}"""
)
classifier_chain=classifier_prompt | llm | StrOutputParser()

#定义路由分支
router_branch=RunnableBranch(
    (lambda x : '川菜' in x['topic'],sichuan_chain),
    (lambda x: '粤菜' in x['topic'] ,cantonese_chain),
    general_chain
)

# 组合完整路由链
full_router_chain={"topic": classifier_chain,"question": lambda x: x['question'] } | router_branch


# 3. 运行
demo_questions = [
    {"question": "麻婆豆腐怎么做？"},      # 应该路由到川菜
    {"question": "白切鸡的正宗做法是什么？"}, # 应该路由到粤菜
    {"question": "番茄炒蛋需要放糖吗？"}      # 应该路由到其他
]
for i ,item in enumerate(demo_questions):
    question=item['question']
    try:
        topic=classifier_chain.invoke({"question":question})
        print(f"路由决策: {topic}")

        #执行完整链
        result=full_router_chain.invoke(item)
        print(f"回答: {result}")
    except Exception as e:
        print(f"执行错误: {e}")

