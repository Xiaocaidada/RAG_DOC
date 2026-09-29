from collections import Counter

from unstructured.partition.auto import partition

file_path="../docs/C2/pdf/rag.pdf"
elements=partition(
    filename=file_path,
    content_type="application/pdf"
)
#打印解析结果
print(f"解析完成，{len(elements)} 元素，{sum(len(str(ele)) for ele in elements)} 个字符")


types=Counter(e.category for e in elements)
print(f"元素类型: {dict(types)}" )

print("\n所有元素")
for i,element in enumerate(elements):
    print(f"Element {i} ({element.category})")
    print(element)
    print("="*60)