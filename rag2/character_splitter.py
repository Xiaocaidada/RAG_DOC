from xml.dom.minidom import CharacterData

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

file_path="../docs/C2/txt/蜂医.txt"
docs=TextLoader(file_path,encoding="utf-8").load()

text_splitter=CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=10
)
chunks=text_splitter.split_documents(docs)
print(f"本文被切分为{len(chunks)} 个块\n")
print("--- 前5个块内容示例 ---")
for i, chunk in enumerate(chunks[:5]):
    print("=" * 60)
    # chunk 是一个 Document 对象，需要访问它的 .page_content 属性来获取文本
    print(f'块 {i+1} (长度: {len(chunk.page_content)}): "{chunk.page_content}"')
