import os
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


class SimpleSQLGenerator:
    def __init__(self,api_key:str=None):
        self.llm= ChatOpenAI(
            model="glm-5.3-flash",
            api_key=api_key,
            base_url="https://open.bigmodel.cn/api/paas/v4",
            temperature=0.1,
            timeout=120,
        )
    def generate_sql(self,user_query:str,knowledge_results:List[Dict[str,Any]]):
        context=self._build_context(knowledge_results)
        #构造提示词
        prompt=f"""你是一个SQL专家。请根据以下信息将用户问题转换为SQL查询语句。

数据库信息:
{context}

用户问题: {user_query}

要求:
1. 只返回SQL语句，不要包含任何解释
2. 确保SQL语法正确
3. 使用上下文提供的表名和字段名
4. 如果需要JOIN，请根据表结构进行合理关联

SQL语句："""
        messages=[HumanMessage(content=prompt)]
        response=self.llm.invoke(messages)

        #清理SQL语句
        fixed_sql=response.content.strip()
        if fixed_sql.startswith("```sql"):
            fixed_sql=fixed_sql[6:]
        if fixed_sql.startswith("```"):
            fixed_sql=fixed_sql[3:]
        if fixed_sql.endswith("```"):
            fixed_sql=fixed_sql[:-3]

        return fixed_sql.strip()

    def _build_context(self,knowledge_results:List[Dict[str,Any]])->str:
        context=""
        ddl_info=[]
        qsql_examples=[]
        descriptions=[]
        for result in knowledge_results:
            if result['type'] == 'ddl':
                ddl_info.append(result['content'])
            elif result['type'] == 'qsql':
                qsql_examples.append(result['content'])
            elif result['type'] == 'description':
                descriptions.append(result['content'])
        if ddl_info:
            context+="=== 表结构信息 ===\n"
            context+="\n".join(ddl_info)+"\n\n"
        if descriptions:
            context+="=== 表和字段描述 ===\n"
            context+="\n".join(descriptions)+"\n\n"
        if qsql_examples:
            context+="===查询示例===\n"
            context+="\n".join(qsql_examples)+"\n\n"
        print(f"从知识库查询到内容: {context}")
        return context




    def fix_sql(self,origin_sql:str,error_message:str,knowledge_results:List[Dict[str,Any]])->str:
        context=self._build_context(knowledge_results)
        prompt=f"""请修复以下SQL语句的错误信息

数据库信息:
{context}

原始SQL
{origin_sql}

错误信息:
{error_message}

请返回修复后的SQL语句(只返回SQL,不要解释):
"""
        messages=[HumanMessage(content=prompt)]
        response=self.llm.invoke(messages)
        fixed_sql=response.content.strip()
        if fixed_sql.startswith("```sql"):
            fixed_sql=fixed_sql[6:]
        if fixed_sql.startswith("```"):
            fixed_sql=fixed_sql[3:]
        if fixed_sql.endswith("```"):
            fixed_sql=fixed_sql[:-3]
        return fixed_sql.strip()



















