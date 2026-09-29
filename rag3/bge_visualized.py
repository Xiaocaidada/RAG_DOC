from visual_bge.visual_bge.modeling import Visualized_BGE
import torch
model=Visualized_BGE(model_name_bge="BAAI/bge-base-en-v1.5",model_weight="../models/bge/Visualized_base_en_v1.5.pth")

model.eval()

with torch.no_grad():
    text_emb=model.encode(text="datawhale开源组织的logo")
    img_emb_1=model.encode(image="../docs/C3/imgs/datawhale01.png")
    multi_emb_1=model.encode(image="../docs/C3/imgs/datawhale01.png",text="datawhale")
    img_emb_2 = model.encode(image="../docs/C3/imgs/datawhale02.png")
    multi_emb_2 = model.encode(image="../docs/C3/imgs/datawhale02.png", text="datawhale开源组织的logo")


sim_1=img_emb_1 @ img_emb_2.T
sim_2=img_emb_1 @ multi_emb_1.T
sim_3=text_emb @ multi_emb_1.T
sim_4=multi_emb_1 @ multi_emb_2.T

print("=== 相似度计算结果 ===")
print(f"纯图像 vs 纯图像: {sim_1}")
print(f"图文结合1 vs 纯图像: {sim_2}")
print(f"图文结合1 vs 纯文本: {sim_3}")
print(f"图文结合1 vs 图文结合2: {sim_4}")

# 向量信息分析
print("\n=== 嵌入向量信息 ===")
print(f"多模态向量维度: {multi_emb_1.shape}")
print(f"图像向量维度: {img_emb_1.shape}")
print(f"多模态向量示例 (前10个元素): {multi_emb_1[0][:10]}")
print(f"图像向量示例 (前10个元素):   {img_emb_1[0][:10]}")
