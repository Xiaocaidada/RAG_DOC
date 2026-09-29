import json
import os
from dataclasses import dataclass
from typing import Dict, Any, List

import cv2
import torch
from PIL import Image
from tqdm.asyncio import tqdm

from visual_bge.modeling import Visualized_BGE

# 放在 import BGEM3EmbeddingFunction 之前！
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import numpy as np
from pymilvus import connections, MilvusClient, FieldSchema, DataType, CollectionSchema, Collection, RRFRanker, \
    AnnSearchRequest
from pymilvus.model.hybrid import BGEM3EmbeddingFunction


@dataclass
class DragonImage:
    img_id:str
    path: str
    title: str
    description: str
    category: str
    location: str
    environment: str
    combat_details: Dict[str,Any] =None
    scene_info: Dict[str,Any]=None


class DragonDataset:
    def __init__(self,data_dir:str,metadata_path:str):
        self.data_dir=data_dir
        self.metadat_path=metadata_path
        self.images:List[DragonImage]=[]
        self._load_metadata()

    def _load_metadata(self):
        with open(self.metadat_path,"r",encoding="utf-8") as f:
            datas=json.load(f)
            for item in datas:
                if not item['path'].startswith(self.data_dir):
                    item['path']=os.path.join(self.data_dir,item['path'])
                self.images.append(DragonImage(**item))

    def get_text_content(self,img:DragonImage):
        parts=[img.title,img.description,img.location,img.environment]
        if img.combat_details:
            parts.extend(img.combat_details.get("combat_style",[]))
            parts.extend(img.combat_details.get("abilities_used",[]))
        if img.scene_info:
            parts.append(img.scene_info.get("time_of_day",''))
        return ' '.join(filter(None,parts))


class HybridMultimodalEncoder:
    def __init__(self,visual_model_name:str,visual_model_path:str):
        self.visual_model=Visualized_BGE(model_name_bge=visual_model_name,model_weight=visual_model_path)
        self.visual_model.eval()

        self.bge_m3=BGEM3EmbeddingFunction(use_fp16=False,device="cpu")
        print(f"BGE-M3 密集向量维度：{self.bge_m3.dim['dense']}")

    def encode_multimodal(self,image_path:str,text:str)->list[float]:
        with torch.no_grad():
            query_emb=self.visual_model.encode(image=image_path,text=text)
        return query_emb.tolist()[0]

    def encode_text_hybrid(self,text:str)->dict:
        embeddings=self.bge_m3([text])
        return {
            'sparse': embeddings['sparse'],
            'dense': embeddings['dense']
        }

    def encode_query(self,image_path:str =None,text:str=None,mode:str="multimodal")->dict:
        result={}
        if mode in ['multimodal','all'] and image_path and text:
            result['multimodal']=self.encode_multimodal(image_path,text)
        if mode in ['hybrid','dense','sparse','all']:
            text_embeddings=self.encode_text_hybrid(text)
            result['dense']=text_embeddings['dense'][0]
            result['sparse']=text_embeddings['sparse'][0]
        return result


def visualize_results(query_image_path: str, retrieved_results: list, search_mode: str,
                      img_height: int = 300, img_width: int = 300, row_count: int = 3) -> np.ndarray:
    """从检索到的结果创建一个全景图用于可视化"""
    panoramic_width = img_width * row_count
    panoramic_height = img_height * row_count
    panoramic_image = np.full((panoramic_height, panoramic_width, 3), 255, dtype=np.uint8)
    query_display_area = np.full((panoramic_height, img_width, 3), 255, dtype=np.uint8)

    # 处理查询图像
    if query_image_path and os.path.exists(query_image_path):
        query_pil = Image.open(query_image_path).convert("RGB")
        query_cv = np.array(query_pil)[:, :, ::-1]
        resized_query = cv2.resize(query_cv, (img_width, img_height))
        bordered_query = cv2.copyMakeBorder(resized_query, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 0, 0))
        query_display_area[img_height * (row_count - 1):, :] = cv2.resize(bordered_query, (img_width, img_height))
        cv2.putText(query_display_area, "Query", (10, panoramic_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 0, 0), 2)
        cv2.putText(query_display_area, search_mode, (10, panoramic_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 100, 0), 2)

    # 处理检索到的图像
    for i, result in enumerate(retrieved_results):
        row, col = i // row_count, i % row_count
        start_row, start_col = row * img_height, col * img_width

        img_path = result['image_path']
        retrieved_pil = Image.open(img_path).convert("RGB")
        retrieved_cv = np.array(retrieved_pil)[:, :, ::-1]
        resized_retrieved = cv2.resize(retrieved_cv, (img_width - 4, img_height - 4))
        bordered_retrieved = cv2.copyMakeBorder(resized_retrieved, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        panoramic_image[start_row:start_row + img_height, start_col:start_col + img_width] = bordered_retrieved

        # 添加索引号和相似度
        cv2.putText(panoramic_image, f"{i + 1}", (start_col + 10, start_row + 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)
        cv2.putText(panoramic_image, f"{result['distance']:.3f}", (start_col + 10, start_row + img_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return np.hstack([query_display_area, panoramic_image])

class HybridMultimodalSearcher:
    def __init__(self,data_dir:str,metadata_path:str,collection_name:str,milvus_uri:str):
        self.data_dir=data_dir
        self.metadata_path=metadata_path
        self.collection_name=collection_name
        self.milvus_uri=milvus_uri

        print("正在初始化数据集")
        self.dataset=DragonDataset(data_dir,metadata_path)
        print(f"加载了 {len(self.dataset.images)} 张龙类图片")

        print("正在初始化混合多模态编码器...")
        self.encoder=HybridMultimodalEncoder(
            visual_model_name="BAAI/bge-base-en-v1.5",
            visual_model_path="../models/bge/Visualized_base_en_v1.5.pth"
        )

        #连接Milvus
        print(f"正在连接到Milvus: {milvus_uri}")
        connections.connect(uri=milvus_uri)
        self.milvus_client=MilvusClient(uri=milvus_uri)
        self.collection=None

    def create_collection(self):
        print(f"正在创建 Collection: {self.collection_name}")
        if self.milvus_client.has_collection(self.collection_name):
            self.milvus_client.drop_collection(self.collection_name)
            print(f"已删除已存在的 Collection: '{self.collection_name}'")
        sample_text=self.dataset.get_text_content(self.dataset.images[0])
        sample_image=self.dataset.images[0].path
        multimodal_dim=len(self.encoder.encode_multimodal(sample_image,sample_text))
        dense_dim=self.encoder.bge_m3.dim['dense']
        print(f"多模态向量维度：{multimodal_dim}")
        print(f"密集向量维度: {dense_dim}")

        fields = [
            FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
            FieldSchema(name="img_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="environment", dtype=DataType.VARCHAR, max_length=64),
            # 三种向量类型
            FieldSchema(name="multimodal_vector", dtype=DataType.FLOAT_VECTOR, dim=multimodal_dim),
            FieldSchema(name="text_sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="text_dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim)
        ]
        schema=CollectionSchema(fields,description="混合多模态龙类图片检索")
        self.collection=Collection(name=self.collection_name,schema=schema,consistency_level="Strong")
        print("Collection 创建成功")

        print("正在创建索引")
        multimodal_index={"index_type":"HNSW","metric_type":"COSINE","params":{"M":16,"efConstruction":256}}
        self.collection.create_index("multimodal_vector",multimodal_index)
        print("多模态向量索引创建成功")

        sparse_index={"index_type":"SPARSE_INVERTED_INDEX","metric_type":"IP"}
        self.collection.create_index("text_sparse_vector",sparse_index)
        print("稀疏向量索引创建成功")

        dense_index={"index_type":"AUTOINDEX","metric_type":"IP"}
        self.collection.create_index("text_dense_vector",dense_index)
        print("密集向量索引创建成功")

        self.collection.load()
        print(f"Collection '{self.collection_name}' 已加载到内存")

    def insert_data(self):
        if self.collection.is_empty:
            print("Collection 为空,开始插入数据")
            img_ids,image_paths,titles,descriptions=[],[],[],[]
            categories,locations,environments=[],[],[]
            multimodal_vectors,text_sparse_vectors,text_dense_vectors=[],[],[]
            for img_data in tqdm(self.dataset.images,desc="生成向量嵌入"):
                text_content=self.dataset.get_text_content(img_data)
                #生成多模态向量
                multimodal_vector=self.encoder.encode_multimodal(img_data.path,text_content)
                #生成文本的混合向量(稀疏+密集)
                text_embeddings=self.encoder.encode_text_hybrid(text_content)
                img_ids.append(img_data.img_id)
                image_paths.append(img_data.path)
                titles.append(img_data.title)
                descriptions.append(img_data.description)
                categories.append(img_data.category)
                locations.append(img_data.location)
                environments.append(img_data.environment)

                multimodal_vectors.append(multimodal_vector)
                text_sparse_vectors.append(text_embeddings['sparse']._getrow(0))
                text_dense_vectors.append(text_embeddings['dense'][0])
            self.collection.insert([
                    img_ids, image_paths, titles, descriptions, categories, locations, environments,
                    multimodal_vectors, text_sparse_vectors, text_dense_vectors
                    ])
            self.collection.flush()
            print(f"数据插入完成，总数: {self.collection.num_entities}")
        else:
            print(f"Collection 中已有 {self.collection.num_entities}条数据，跳过插入")

    def search(self,query_image:str ,query_text:str,mode:str="hybrid",top_k=5):
        search_params={"metric_type":"IP","params":{}}
        cosine_params={"metric_type":"COSINE","params":{"ef":128}}
        output_fields = ["img_id", "image_path", "title", "description", "category", "location", "environment"]
        if mode=="multimodal":
            query_vector=self.encoder.encode_multimodal(query_image,query_text)
            results=self.collection.search(
                [query_vector],"multimodal_vector",param=cosine_params,
                limit=top_k,output_fields=output_fields)[0]
        elif mode=="dense":
            query_embeddings=self.encoder.encode_text_hybrid(query_text)
            dense_vec=query_embeddings['dense'][0]
            results=self.collection.search(
                [dense_vec],"text_dense_vector",param=search_params,
                limit=top_k,output_fields=output_fields
            )[0]
        elif mode=="sparse":
            query_embeddings=self.encoder.encode_text_hybrid(query_text)
            sparse_vec=query_embeddings['sparse']._getrow(0)
            results=self.collection.search(
                [sparse_vec],"text_sparse_vector",param=search_params,
                limit=top_k,output_fields=output_fields
            )[0]
        elif mode=='hybrid':
            query_embeddings=self.encoder.encode_text_hybrid(query_text)
            dense_vec=query_embeddings['dense'][0]
            sparse_vec=query_embeddings['sparse']._getrow(0)
            #创建RRF融合器
            rerank=RRFRanker(k=60)
            dense_req=AnnSearchRequest([dense_vec],"text_dense_vector",search_params,limit=top_k)
            sparse_req=AnnSearchRequest([sparse_vec],"text_sparse_vector",search_params,limit=top_k)
            results=self.collection.hybrid_search(
                [dense_req,sparse_req],rerank=rerank,limit=top_k,output_fields=output_fields
            )[0]
        return results

    def compare_search_modes(self,query_image_path:str,query_text:str,top_k=5):
        modes=["multimodal","dense","sparse","hybrid"]
        results={}
        print(f"\n{'=' * 50}")
        print(f"查询图像: {query_image_path}")
        print(f"查询文本: {query_text}")
        print(f"{'=' * 50}")

        for mode in modes:
            print(f"\n[{mode.upper()}] 搜索结果")
            search_results=self.search(query_image_path,query_text,mode,top_k)
            mode_results=[]
            for i,hit in enumerate(search_results):
                print(f"{i+1}.{hit.entity.get("title")} (Score: {hit.distance:.4f})")
                print(f"路径: {hit.entity.get('image_path')}")
                print(f"描述: {hit.entity.get('description')[:80]}...")
                mode_results.append({
                    "image_path":hit.entity.get("image_path"),
                    "distance":hit.distance,
                    "title":hit.entity.get("title")
                })
            results[mode]=mode_results
        return results

    def visualize_comparison(self,query_image_path:str,query_text:str,top_k=3):
        modes = ["multimodal", "dense", "sparse", "hybrid"]
        for mode in modes:
            results=self.search(query_image_path,query_text,mode,top_k)
            retrieved_results=[]
            for hit in results:
                retrieved_results.append({
                    "image_path":hit.entity.get("image_path"),
                    "distance":hit.distance
                })
            if retrieved_results:
                panoramic_image=visualize_results(query_image_path,retrieved_results,mode.upper())
                output_path=f"../docs/C4/{mode}_search_result.png"
                cv2.imwrite(output_path,panoramic_image)
                print(f"{mode.upper()} 搜索结果已保存到: {output_path}")

    def cleanup(self):
        if self.collection:
            self.collection.release()
            print(f"已从内存中释放Collection: '{self.collection_name}'")
            self.milvus_client.drop_collection(self.collection_name)
            print(f"已删除COllection: '{self.collection_name}'")


if __name__ == "__main__":
    COLLECTION_NAME = "dragon_hybrid_demo"
    MILVUS_URI = "http://192.168.148.3:19530"
    DATA_PATH = "../docs/C3/dragon"
    METADATA_PATH="../docs/C4/metadata/dragon.json"

    searcher=HybridMultimodalSearcher(DATA_PATH,METADATA_PATH,COLLECTION_NAME,MILVUS_URI)

    try:
        #创建集合并且插入数据
        searcher.create_collection()
        searcher.insert_data()

        query_image_path=os.path.join(DATA_PATH,"query.png")
        query_text="悬崖上的巨龙"

        #对比不同搜索结果模式
        searcher.compare_search_modes(query_image_path,query_text,top_k=3)
        #可视化结果
        searcher.visualize_comparison(query_image_path,query_text,top_k=3)
        print(f"\n{'=' * 50}")
        print("搜索模式分析:")
        print("- MULTIMODAL: 结合图像和文本的多模态向量检索")
        print("- DENSE: 基于语义的密集向量检索")
        print("- SPARSE: 基于关键词的稀疏向量检索")
        print("- HYBRID: 稀疏+密集向量RRF融合检索")
        print(f"{'=' * 50}")
    finally:
        searcher.cleanup()
