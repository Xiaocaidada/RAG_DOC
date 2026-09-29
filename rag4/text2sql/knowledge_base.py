import json
import os
from typing import Dict, List, Any

from pymilvus import MilvusClient, FieldSchema, DataType, Collection, CollectionSchema
from pymilvus.model.hybrid import BGEM3EmbeddingFunction


class SimpleKnowledgeBase:

    def __init__(self,milvus_uri:str="http://192.168.148.3:19530"):
        self.milvus_uri=milvus_uri
        self.client=MilvusClient(uri=self.milvus_uri)
        self.embedding_function=BGEM3EmbeddingFunction(use_fp16=False,device="cpu")
        self.collection_name="text2sql_kb"
        self._setup_collection()

    def _setup_collection(self):
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
        fields=[
            FieldSchema(name="pk",dtype=DataType.VARCHAR,is_primary=True,auto_id=True,max_length=100),
            FieldSchema(name="content",dtype=DataType.VARCHAR,max_length=4096),
            FieldSchema(name="type",dtype=DataType.VARCHAR,max_length=32),
            FieldSchema(name="dense_vector",dtype=DataType.FLOAT_VECTOR,dim=self.embedding_function.dim['dense'])
        ]
        schema=CollectionSchema(fields,description="Text2SQL知识库")
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            consistency_level="Strong"
        )
        index_params=self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_type="AUTOINDEX",
            metric_type="IP"
        )
        self.client.create_index(collection_name=self.collection_name,index_params=index_params)


    def load_data(self):
        data_dir=os.path.join(os.path.dirname(__file__),"data")
        # 加载ddl数据
        ddl_path=os.path.join(data_dir,"ddl_examples.json")
        if os.path.exists(ddl_path):
            with open(ddl_path,encoding="utf-8") as f:
                ddl_data=json.load(f)
            self._add_ddl_data(ddl_data)
        # 加载q->sql数据
        qsql_path=os.path.join(data_dir,"qsql_examples.json")
        if os.path.exists(qsql_path):
            with open(qsql_path,encoding="utf-8") as f:
                qsql_data=json.load(f)
            self._add_qsql_data(qsql_data)
        # 加载描述数据
        desc_path=os.path.join(data_dir,"db_descriptions.json")
        if os.path.exists(desc_path):
            with open(desc_path,encoding="utf-8") as f:
                desc_data=json.load(f)
            self._add_description_data(desc_data)
        #加载集合到内存
        self.client.load_collection(collection_name=self.collection_name)
        print("知识库加载完成")

    def _add_ddl_data(self,data: List[Dict]):
        contents=[]
        types=[]
        for item in data:
            content=f"表名: {item.get('table_name','')}\n"
            content+=f"DDL: {item.get('ddl_statement','')}\n"
            content+=f"描述: {item.get('description','')}"
            contents.append(content)
            types.append("ddl")
        self._insert_data(contents,types)

    def _add_qsql_data(self,data:List[Dict]):
        contents=[]
        types=[]
        for item in data:
            content=f"问题: {item.get('question','')}\n"
            content+=f"SQL: {item.get('sql','')}"

            contents.append(content)
            types.append("qsql")

        self._insert_data(contents,types)

    def _add_description_data(self,data:List[Dict]):
        contents=[]
        types=[]
        for item in data:
            content=f"表名: {item.get('table_name','')}\n"
            content+=f"表描述: {item.get('table_description','')}\n"
            columns=item.get('columns',[])
            if columns:
                for col in columns:
                    content+=f"  - {col.get('name','')}: {col.get('description','')} ({col.get('type','')})\n"
            contents.append(content)
            types.append("description")

        self._insert_data(contents,types)

    def _insert_data(self,contents:List[str],types:List[str]):
        if not contents:
            return
        embeddings=self.embedding_function(contents)
        data_to_insert=[]
        for i in range(len(contents)):
            data_to_insert.append({
                "content":contents[i],
                "type":types[i],
                "dense_vector": embeddings['dense'][i]
            })
        self.client.insert(collection_name=self.collection_name,data=data_to_insert)

    def search(self,query:str,top_k:int=5)->List[Dict[str,Any]]:
        self.client.load_collection(collection_name=self.collection_name)
        query_embeddings=self.embedding_function([query])
        search_results=self.client.search(
            collection_name=self.collection_name,
            data=query_embeddings['dense'],
            anns_field="dense_vector",
            search_params={"metric_type":"IP"},
            limit=top_k,
            output_fields=["content","type"]
        )[0]
        results=[]
        for hit in search_results:
            results.append({
                "content": hit['entity']['content'],
                "type": hit['entity']['type'],
                "score": hit['distance']
            })
        return results

    def cleanup(self):
        try:
            self.client.drop_collection(self.collection_name)
        except:
            pass

