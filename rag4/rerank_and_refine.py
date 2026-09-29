import os
# 国内hf镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 关闭软链接警告（可选）
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from typing import Sequence

import torch
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, DocumentCompressorPipeline
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

class ColBERTReranker(BaseDocumentCompressor):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

        model_name="bert-base-uncased"
        #加载模型和分词器
        object.__setattr__(self,'tokenizer',AutoTokenizer.from_pretrained(model_name))
        object.__setattr__(self, 'model', AutoModel.from_pretrained(model_name, from_tf=True))

        self.model.eval()
        print(f"ColBERT模型加载完成")

    def encode_text(self,texts):
        inputs=self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        with torch.no_grad():
            outputs=self.model(**inputs)
        embeddings=outputs.last_hidden_state
        embeddings=F.normalize(embeddings,p=2,dim=-1)
        return embeddings

    def calculate_colbert_similarity(self,query_emb,doc_embs,query_mask,doc_masks):
        scores=[]
        for i ,doc_emb in enumerate(doc_embs):
            doc_mask=doc_masks[i:i+1]
            similarity_matrix=torch.matmul(query_emb,doc_emb.unsqueeze(0).transpose(-2,-1))
            doc_mask_expanded=doc_mask.unsqueeze(1)
            similarity_matrix=similarity_matrix.masked_fill(~doc_mask_expanded.bool(),-1e9)
            max_sim_per_query_token=similarity_matrix.max(dim=-1)[0]
            query_mask_expaned=query_mask.unsqueeze(0)
            max_sim_per_query_token=max_sim_per_query_token.masked_fill(~query_mask_expaned.bool(),0)
            colbert_score=max_sim_per_query_token.sum(dim=-1).item()
            scores.append(colbert_score)
        return scores


    def compress_documents(self,documents:Sequence[Document],query:str,callbacks=None)->Sequence[Document]:
        if len(documents) ==0:
            return documents
        query_inputs=self.tokenizer([query],
                       return_tensors="pt",
                       padding=True,
                       truncation=True,
                       max_length=128)
        with torch.no_grad():
            query_outputs=self.model(**query_inputs)
            query_embeddings=F.normalize(query_outputs.last_hidden_state,p=2,dim=-1)
        doc_texts=[doc.page_content for doc in documents]
        doc_inputs=self.tokenizer(
            doc_texts,
            padding=True,
            truncation=True,
            max_length=128
        )
        with torch.no_grad():
            doc_outputs=self.model(**doc_inputs)
            doc_embeddings=F.normalize(doc_outputs.last_hidden_state,p=2,dim=-1)

        scores=self.calculate_colbert_similarity(
            query_embeddings,doc_embeddings,query_inputs['attention_mask'],doc_inputs['attention_mask']
        )
        scored_docs=list(zip(documents,scores))
        scored_docs.sort(key=lambda x : x[1],reverse=True)
        reranked_docs=[doc for doc,_ in scored_docs[:5]]
        return reranked_docs


hf_bge_embeddings=HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
llm=ChatOpenAI(
    model="glm-5.3-flash",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    api_key=os.getenv("ZAI_API_KEY"),
    temperature=0.6,
)

doc=TextLoader("../docs/C4/txt/ai.txt",encoding="utf-8").load()
splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
docs=splitter.split_documents(doc)
vectorstore=FAISS.from_documents(docs,hf_bge_embeddings)
base_retriever=vectorstore.as_retriever()

reranker=ColBERTReranker()
compressor=LLMChainExtractor.from_llm(llm)

pipeline_compressor=DocumentCompressorPipeline(transformers=[reranker,compressor])

# 创建包装类
final_retriever=ContextualCompressionRetriever(
    base_compressor=pipeline_compressor,
    base_retriever=base_retriever
)

# 7. 执行查询并展示结果
query = "AI还有哪些缺陷需要克服？"
print(f"\n{'='*20} 开始执行查询 {'='*20}")
print(f"查询: {query}\n")



# 7.1 基础检索结果
print(f"--- (1) 基础检索结果 (Top 20) ---")


# 7.1 基础检索结果
print(f"--- (1) 基础检索结果 (Top 20) ---")
base_results = base_retriever.get_relevant_documents(query)
for i, doc in enumerate(base_results):
    print(f"  [{i+1}] {doc.page_content[:100]}...\n")



# 7.2 使用管道压缩器的最终结果
print(f"\n--- (2) 管道压缩后结果 (ColBERT重排 + LLM压缩) ---")
final_results=final_retriever.get_relevant_documents(query)
for i, doc in enumerate(final_results):
    print(f"  [{i+1}] {doc.page_content}\n")
