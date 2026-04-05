# BLIP2 多模态商品类目识别 — 架构设计

## 项目目标
基于BLIP2的Zero-shot能力实现商品类目识别，三阶段推理框架。

## 目标指标
| 版本 | 叶子类目TOP1 | 二级类目TOP1 | 一级类目TOP1 |
|------|-------------|-------------|-------------|
| V1   | 72.0%       | 82.4%       | 88.5%       |
| V2 (LoRA) | +3.2%  | +4.0%       | +5.6%       |

## 数据方案

### 当前: 合成数据 (用于流程验证)
- 一级类目: 11个, 二级: 55个, 叶子: 550个
- 总商品: 22000条, 锚/测试: 70%/30%
- 图片: 随机颜色方块 (无视觉语义)
- 状态: 流程已跑通, 指标低(预期行为)

### 计划: 真实数据集替换

**首选: Products-10K (京东)**
- 来源: JD.com, Kaggle可直接下载
- 规模: ~150K图片, 10,000 SKU级类目
- 层级: 有层级结构 (时尚/3C/食品/家居等)
- 文本: 无 (需用类目名称生成或补充)
- 大小: 19.6GB
- URL: https://www.kaggle.com/datasets/warcoder/visual-product-recognition

**备选:**
| 数据集 | 规模 | 特点 | URL |
|--------|------|------|-----|
| AliProducts2 | 400万图文对, 5万类 | 最大规模, 含文本, 需天池注册 | tianchi.aliyun.com |
| RP2K | 50万图, 2000 SKU | 中国零售场景, Kaggle可下 | kaggle.com/datasets/khyeh0719/rp2k-dataset |
| Fashion Product Images | 4.4万图 | 层级+文本, 英文, MIT协议 | kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset |

**数据适配策略:**
1. 下载Products-10K, 构建层级类目映射 (SKU → L2 → L1)
2. 用类目名称作为文本描述 (类似当前的category text template)
3. 按70/30拆分anchor/test
4. 复用现有pipeline代码, 只需改data loader

## 系统架构
```
离线阶段:
  锚商品(文本+图片) → BLIP2 Q-Former → 多模态Embedding → FAISS索引
  类目文本描述      → BLIP2 Q-Former → 文本Embedding   → 存储

在线阶段:
  新商品 → Embedding提取
       ↓
  [召回] 多模态Embedding vs 锚商品Embedding → TopN锚商品 → 加权候选类目
       ↓
  [排序] 图片Embedding vs 类目文本Embedding → TopN叶子类目
       ↓
  [重排] 融合召回+排序分数 → Top1类目预测
```

## 核心配置参数
- 召回: top_n=50, text_weight=0.4, image_weight=0.6
- 排序: top_n=10
- 重排: recall_weight=0.45, rank_weight=0.55

## V2升级方向
- LoRA微调Q-Former，提升domain-specific特征提取
- 训练数据: 使用anchor products作为训练集
- 预期提升: 叶子+3.2%, 二级+4.0%, 一级+5.6%

## 技术栈
- 模型: Salesforce/blip2-opt-2.7b
- 向量检索: FAISS (IndexFlatIP)
- 框架: PyTorch + Transformers + PEFT(LoRA)

## 技术备忘
- transformers 5.x 移除了 `get_text_features` 和 `get_qformer_features`
- Q-Former cross-attention 层必须有 encoder_hidden_states (不支持 text-only)
- text embedding 使用 LM word_embeddings, image embedding 使用 Q-Former + language_projection
- 两者在同一 LM embedding space (dim=2560), 可做 cosine similarity
