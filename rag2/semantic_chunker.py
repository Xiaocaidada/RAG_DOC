from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

embeddings=HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device":"cpu"},
    encode_kwargs={"normalize_embeddings":True}
)


text_splitter=SemanticChunker(embeddings,breakpoint_threshold_type="percentile")
docs=TextLoader("../docs/C2/txt/蜂医.txt",encoding="utf-8").load()
chunks=text_splitter.split_documents(docs)
print(f"文本被切分为 {len(chunks)} 个块。\n")
print("--- 前2个块内容示例 ---")
for i, chunk in enumerate(chunks[:2]):
    print("=" * 60)
    print(f'块 {i+1} (长度: {len(chunk.page_content)}):\n"{chunk.page_content}"')
