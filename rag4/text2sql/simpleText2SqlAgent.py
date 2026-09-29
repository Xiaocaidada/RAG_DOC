import os
import sqlite3
from typing import Dict, Any, Tuple, List

from rag4.text2sql.knowledge_base import SimpleKnowledgeBase
from rag4.text2sql.simpleSQLGenerator import SimpleSQLGenerator



class SimpleText2Agent:

    def __init__(self,milvus_uri:str="http://192.168.148.3:19530",api_key:str=None):
        self.knowledge_base=SimpleKnowledgeBase(milvus_uri=milvus_uri)
        self.sql_generator=SimpleSQLGenerator(api_key)
        self.db_path=None
        self.connection=None

        self.max_retry_count=3
        self.top_k_retrieval=5
        self.max_result_rows=100


    def connect_database(self,db_path:str)->bool:
        try:
            self.db_path=db_path
            self.connection=sqlite3.connect(db_path)
            print(f"成功连接到数据库: {db_path}")
            return True
        except Exception as e:
            print(f"数据库连接失败: {db_path}")
            return False

    def load_knowledge_base(self):
        self.knowledge_base.load_data()


    def query(self,user_question:str) ->Dict[str,Any]:
        if not self.connection:
            return {
                "success": False,
                "error" : "数据库未连接",
                "sql": None,
                "results":None
            }
        print(f"\n=== 处理查询: {user_question} ===")

        # 从知识库检索
        knowledge_results=self.knowledge_base.search(user_question)
        print(f"检索到{len(knowledge_results)} 条相关信息")

        # 生成SQL
        sql=self.sql_generator.generate_sql(user_question,knowledge_results)
        print(f"生成的SQL: {sql}")

        # 执行SQL(带重试)
        retry_count=0

        while retry_count<self.max_retry_count:
            print(f"执行SQL (尝试: {retry_count+1}/ {self.max_retry_count})...")
            success,result=self._execute_sql(sql)

            if success:
                print("SQL执行成功!")
                return {
                    "success": True,
                    "error": None,
                    "sql":sql,
                    "results": result,
                    "retry_count": retry_count+1
                }
            else:
                print(f"SQL执行失败: {result}")
                if retry_count<self.max_retry_count-1:
                    print("尝试修复SQL...")
                    sql=self.sql_generator.fix_sql(sql,result,knowledge_results)
                    print(f"修复后的SQL: {sql}")
                retry_count+=1
        return{
            "success":False,
            "errror":f"超过最大重试次数 ({self.max_retry_count})",
            "sql":sql,
            "results": None,
            "retry_count":retry_count
        }

    def _execute_sql(self,sql:str)->Tuple[bool,Any]:
        try:
            cursor=self.connection.cursor()
            if sql.strip().upper().startswith('SELECT') and 'LIMIT' not in sql.upper():
                sql= f"{sql.rstrip(';')} LIMIT {self.max_result_rows}"
            cursor.execute(sql)

            if sql.strip().upper().startswith('SELECT'):
                columns=[desc[0] for desc in cursor.description]
                rows=cursor.fetchall()
                results=[]
                for row in rows:
                    result_row={}
                    for i, value in enumerate(row):
                        result_row[columns[i]]=value
                    results.append(result_row)
                cursor.close()
                return True,{
                    "columns":columns,
                    "rows": results,
                    "count": len(results)
                }
            else:
                self.connection.commit()
                cursor.close()
                return True,"SQL执行成功"
        except Exception as e:
            return False,e

    def add_example(self,question:str,sql:str):
        data_dir=os.path.join(os.path.dirname(__file__),"data")
        qsql_path=os.path.join(data_dir,"qsql_examples.json")
        try:
            import json
            if os.path.exists(qsql_path):
                with open(qsql_path) as f:
                    data=json.load(f)
            else:
                data=[]

            data.append({
                "question":question,
                "sql":sql,
                "database":"sqlite"
            })

            with open(qsql_path,'w',encoding="utf-8") as f:
                json.dump(data,f,ensure_ascii=False,indent=2)

            print(f"已添加新示例: {question}")
        except Exception as e:
            print(f"添加示例失败: {question}")

    def get_table_info(self) -> List[Dict[str,Any]]:
        if not self.connection:
            return []
        try:
            cursor=self.connection.cursor()
            cursor.execute("SELECT name from sqlite_master WHERE type= 'table'")
            tables=cursor.fetchall()
            table_info=[]
            for table in tables:
                table_name=table[0]
                cursor.execute(f"PRAGMA table_info ({table_name})")
                columns=cursor.fetchall()

                table_info.append({
                    "table_name":table_name,
                    "columns":[{
                        "name":col[1],
                        "type":col[2],
                        "nullable":col[3],
                        "default": col[4],
                        "primary_key":bool(col[5])
                    } for col in columns]
                })
            cursor.close()
            return table_info
        except Exception as e:
            print(f"获取表信息失败: {str(e)}")

    def cleanup(self):
        if self.connection:
            self.connection.close()
            self.connection=None
            print("数据库连接已关闭")
        self.knowledge_base.cleanup()
        print("知识库已清理")







