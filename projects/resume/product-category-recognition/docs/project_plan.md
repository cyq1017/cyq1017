# 项目计划 — 多模态商品类目识别

## 阶段目标

### 阶段1: V0 — 资料梳理 [已完成]
- 阅读现有文档，梳理技术架构
- 理解三阶段推理具体过程
- 初始化git仓库

### 阶段2: V1 — 代码重建 [进行中]
- [x] BLIP2 zero-shot离线推理模块
- [x] 召回模块（多模态embedding + FAISS）
- [x] 排序模块（图片vs类目文本embedding）
- [x] 重排模块（分数融合）
- [x] 单元测试 (23个, 全通过)
- [x] 合成数据验证流程跑通
- [x] Products-10K真实数据集下载+适配
- [ ] 真实数据评测验证指标
- [ ] git tag v1.0

### 阶段3: V2 — LoRA微调
- [ ] LoRA微调方案设计 (走ECC brainstorming → plan)
- [ ] Q-Former LoRA微调（self-attention + cross-attention）
- [ ] 指标验证 & git tag v2.0

## 技术决策
- **模型**: Salesforce/blip2-opt-2.7b (Q-Former + OPT-2.7B)
- **向量检索**: FAISS IndexFlatIP (精确内积搜索)
- **数据**: Products-10K (京东, 141K图片, 9691 SKU)
- **text embedding**: LM word_embeddings (因Q-Former不支持text-only)
- **兼容性**: `from __future__ import annotations` 支持Python 3.9

## 架构设计
详见 docs/design.md
