# V2 LoRA Fine-tuning Design Spec

**Date:** 2026-04-08
**Status:** Draft (post-review revision)
**Baseline:** V1 tag v1.0, leaf_top1=24.6% (Products-10K, 9691 SKU, zero-shot)

## Goal

LoRA fine-tune BLIP2's Q-Former to improve product category recognition accuracy.
Reproduce the approach described in the resume: LoRA on Q-Former self-attention + cross-attention.

## Data Preparation

### 1. Category Subset Selection
- From Products-10K (9691 SKU, 141K images), select **top ~500 highest-frequency leaf categories**
- Target: ~20,000 products total (matching resume's "2w+ 商品样本, 500+ 叶子类目")
- Split: 70% train / 30% test (stratified by leaf category)

### 2. LLM-Generated Category Descriptions
- For each leaf category, use LLM (MiniMax API, OpenAI-compatible) to generate:
  - Extended category description (hierarchy + typical product types)
  - Representative brand/keyword list
- Template prompt: "给出'{l1}/{l2}/{leaf}'类目的详细描述，包括典型品类词、常见品牌、商品特征"
- API: MiniMax via `openai` SDK, base_url=`https://api.minimax.io/v1`
- Replaces the simple category name used in V1
- **Cache:** generated descriptions saved to `data/category_descriptions.json`, only regenerate missing entries
- **Fallback:** if API fails, use V1 template "{l1} {l2} {leaf}"
- Estimated: ~500 API calls, ~200 tokens each

### 3. Training Data Format
- Each sample: (product_image, category_description_text, leaf_label)
- Positive pair: image matched with its own category description
- Negative pairs: in-batch negatives (standard ITC approach)

## Fine-tuning Architecture

### LoRA Configuration
- **Target modules:** Q-Former self-attention (q, k, v) + cross-attention (q, k, v)
- **LoRA rank:** 8 (default, will experiment with 4/16)
- **LoRA alpha:** 16 (2x rank)
- **LoRA dropout:** 0.1
- **Trainable params:** ~2% of Q-Former (~2.4M / 107M params)

### Training Objective
- **Image-Text Contrastive (ITC) loss**
- **CRITICAL: Text embedding path must match V1 inference**
  - V1 inference uses `LM.get_input_embeddings()` mean-pool for text (not Q-Former BERT encoder)
  - ITC training text side MUST use the same `LM.get_input_embeddings()` path
  - This ensures LoRA-adapted Q-Former image outputs align to the same text space used at inference
- Image side: Q-Former → language_projection → [D=2560]
- Text side: LM word_embeddings → mean-pool → [D=2560] (frozen, same as V1)
- Loss: cosine similarity based contrastive loss with in-batch negatives
- Effective batch size: 32 (batch_size=16 × gradient_accumulation=2)
- Why ITC: directly optimizes the cosine similarity used in V1 pipeline

### Training Hyperparameters
```yaml
training:
  epochs: 3           # start small, increase if underfitting
  batch_size: 16      # fits RTX 5090 32GB with LoRA
  learning_rate: 1e-4
  weight_decay: 0.01
  warmup_ratio: 0.1
  scheduler: cosine
  fp16: true
  gradient_accumulation_steps: 2
```

### Post-training Experiment
- Compare rank=4 vs 8 vs 16
- Record: trainable params, training time, final loss, eval metrics
- Log to docs/devlog.md

## Model Persistence

### Save
- LoRA adapter saved via `model.save_pretrained("checkpoints/v2-lora-rank8/")`
- Only adapter weights (~10MB), not full model
- Save after each epoch for early stopping

### Load at Inference
- `BLIP2Encoder` updated to accept optional `lora_adapter_path` parameter
- At init: load base model → `PeftModel.from_pretrained(model, adapter_path)` → merge or keep separate
- Merge option: `model.merge_and_unload()` for zero overhead at inference

### Re-embedding After Training
- **Anchor embeddings: must regenerate** (image path goes through LoRA-adapted Q-Former)
- **Category text embeddings: no change** (text path uses LM word_embeddings, unaffected by LoRA)
- **FAISS index: must rebuild** from new anchor embeddings
- Estimated re-embedding time: ~10 min for 20K subset on RTX 5090

## Evaluation

### Success Criteria
- V2 leaf_top1 on 500-category subset > V1 zero-shot on same subset
- Target improvement: +5-10% (absolute) on leaf_top1
- Note: 500 classes is easier than 9691, so absolute numbers not directly comparable to V1 full baseline

### Process
- First: run V1 zero-shot on 500-category subset → establish comparable baseline
- Then: run V2 LoRA on same subset → compare
- Metrics: TopK accuracy at leaf/L2/L1 levels (K=1,3,5)
- Same pipeline: offline embedding → FAISS index → recall → ranking → rerank

### Data Split Integrity
- Train set (70%) = anchor set for FAISS index (same products)
- Test set (30%) = evaluation set (strictly held out, never in FAISS index)
- No overlap between train/anchor and test sets

## File Structure (New)

```
src/train/
  ├── __init__.py
  ├── dataset.py           # ITC training dataset (image + LLM-generated text)
  ├── lora_config.py       # LoRA configuration for Q-Former
  ├── trainer.py           # Training loop with ITC loss
  └── train_utils.py       # Utilities (logging, checkpointing)
scripts/
  ├── prepare_v2_data.py   # Select top 500 categories, build train/test split
  ├── generate_descriptions.py  # LLM category description generation
  └── run_train.py         # Training entry point
config/
  └── train.yaml           # Training hyperparameters
```

## Execution Plan

1. **Data preparation** (no GPU needed)
   - Select top 500 categories from Products-10K
   - Generate LLM category descriptions via MiniMax API
   - Build train/test CSV files

2. **Training code** (no GPU needed for writing/testing)
   - Implement ITC dataset, LoRA config, trainer
   - Unit tests with mock data

3. **Training run** (GPU needed)
   - First: 20K subset, 3 epochs (~60-90 min) — validate loss decreases
   - Then: full training, evaluate metrics

4. **Evaluation** (GPU needed)
   - Re-run offline embedding with LoRA-adapted model
   - Run full pipeline evaluation
   - Compare V1 vs V2 metrics

5. **Experiments** (GPU needed)
   - rank=4/8/16 comparison
   - Record results

6. **Wrap-up**
   - Update docs/design.md with V2 results
   - commit + tag v2.0

## Known Issues from V1

- `get_multimodal_embeddings_batch` ignores text parameter (image-only) — to be fixed in V2
- Pseudo-hierarchy (L1/L2) is random grouping, not semantic — acceptable for training, note in reporting

## Dependencies

- peft==0.14.0 (pin version for transformers 5.x compatibility)
- openai SDK (for MiniMax API)
- User provides: MiniMax API key

## Risks

- **Q-Former LoRA target module names** may differ across transformers versions — inspect `model.qformer.named_modules()` on GPU server before writing LoRA config
- **fp16 + LoRA dtype mismatch**: LoRA adapters default to fp32, base model is fp16 — may need explicit dtype handling
- **500-category subset** changes task difficulty vs full 9691 — V1/V2 comparison must use same subset
- **LLM description quality**: niche categories may get hallucinated descriptions — spot-check a sample before full training
- **No early stopping**: if loss plateaus or diverges, currently no automatic stop — add manual checkpoint review after epoch 1

---

# V2 LoRA 微调设计方案（中文版）

**日期:** 2026-04-08
**状态:** 草稿（经 review 修订）
**基线:** V1 tag v1.0, 叶子TOP1=24.6% (Products-10K, 9691 SKU, zero-shot)

## 目标

对 BLIP2 的 Q-Former 做 LoRA 微调，提升商品类目识别准确率。复现简历描述的方案。

## 数据准备

### 1. 类目子集筛选
- 从 Products-10K（9691 SKU, 14.1万图片）中选 **top ~500 高频叶子类目**
- 目标：约 **2万商品**（对应简历"2w+商品样本, 500+叶子类目"）
- 拆分：70% 训练 / 30% 测试（按叶子类目分层）

### 2. LLM 生成类目描述
- 用 LLM（MiniMax API）为每个叶子类目生成：
  - 丰富的类目描述（层级结构 + 典型品类）
  - 代表性品牌/关键词列表
- 提示词模板："给出'{一级}/{二级}/{叶子}'类目的详细描述，包括典型品类词、常见品牌、商品特征"
- API：MiniMax，通过 openai SDK 调用
- **缓存：** 生成结果保存到 `data/category_descriptions.json`，只补生缺失条目
- **兜底：** API 失败时用 V1 模板 "{一级} {二级} {叶子}"
- 预估：~500 次 API 调用，每次 ~200 token

### 3. 训练数据格式
- 每个样本：(商品图片, 类目描述文本, 叶子标签)
- 正样本对：图片与其所属类目的描述
- 负样本：batch 内其他样本（标准 ITC 做法）

## 微调架构

### LoRA 配置
- **目标模块：** Q-Former self-attention (q, k, v) + cross-attention (q, k, v)
- **LoRA rank：** 8（默认值，后续实验 4/16）
- **LoRA alpha：** 16（2倍 rank）
- **LoRA dropout：** 0.1
- **可训练参数：** Q-Former 总参数的 ~2%（约 240万 / 1.07亿）

### 训练目标
- **图文对比学习（ITC）loss**
- **关键：训练时的 text embedding 路径必须和 V1 推理一致**
  - V1 推理用 `LM.get_input_embeddings()` mean-pool 生成 text embedding
  - ITC 训练的 text 侧**必须用同一路径**
  - 确保 LoRA 微调后的 Q-Former 图像输出，对齐到推理时使用的同一 text 空间
- Image 侧：Q-Former → language_projection → [D=2560]
- Text 侧：LM word_embeddings → mean-pool → [D=2560]（冻结，和 V1 一致）
- Loss：基于余弦相似度的对比 loss，batch 内负采样
- 有效 batch size：32（batch_size=16 × 梯度累积=2）

### 训练超参
| 参数 | 值 |
|------|-----|
| epochs | 3（先跑通，欠拟合再加）|
| batch_size | 16 |
| learning_rate | 1e-4 |
| weight_decay | 0.01 |
| warmup_ratio | 0.1 |
| scheduler | cosine |
| fp16 | true |
| 梯度累积 | 2 |

### 训练后实验
- 对比 rank=4 / 8 / 16
- 记录：可训练参数量、训练时间、最终 loss、评测指标
- 写入 docs/devlog.md

## 模型持久化

### 保存
- LoRA adapter 通过 `model.save_pretrained("checkpoints/v2-lora-rank8/")` 保存
- 只保存 adapter 权重（~10MB），不保存完整模型
- 每个 epoch 保存一次，支持早停

### 推理加载
- `BLIP2Encoder` 新增可选参数 `lora_adapter_path`
- 初始化：加载基座模型 → `PeftModel.from_pretrained(model, adapter_path)` → 合并或保持分离
- 合并选项：`model.merge_and_unload()` 推理零额外开销

### 训练后重新生成 Embedding
- **锚商品 embedding：必须重新生成**（图片路径经过 LoRA 微调的 Q-Former）
- **类目文本 embedding：无需变化**（text 路径用 LM word_embeddings，不受 LoRA 影响）
- **FAISS 索引：必须重建**（基于新的锚商品 embedding）
- 预估重新生成时间：RTX 5090 上约 10 分钟（2万子集）

## 评测

### 成功标准
- V2 在 500 类目子集上的 leaf_top1 > V1 zero-shot 在同一子集上的结果
- 目标提升：leaf_top1 绝对值 +5~10%
- 注意：500 类比 9691 类简单，绝对数值不可直接与 V1 全量基线对比

### 流程
1. 先在 500 类目子集上跑 V1 zero-shot → 建立可比基线
2. 再在同一子集上跑 V2 LoRA → 对比
3. 指标：TopK 准确率（leaf/L2/L1, K=1,3,5）
4. 同一 pipeline：离线 embedding → FAISS 索引 → 召回 → 排序 → 重排

### 数据隔离
- 训练集（70%）= FAISS 索引的锚商品（同一批商品）
- 测试集（30%）= 评测集（严格隔离，不出现在 FAISS 索引中）

## 新增文件结构

```
src/train/
  ├── __init__.py
  ├── dataset.py           # ITC 训练数据集
  ├── lora_config.py       # Q-Former LoRA 配置
  ├── trainer.py           # 训练循环 + ITC loss
  └── train_utils.py       # 工具函数（日志、checkpoint）
scripts/
  ├── prepare_v2_data.py   # 筛选 top 500 类目，构建 train/test
  ├── generate_descriptions.py  # LLM 生成类目描述
  └── run_train.py         # 训练入口
config/
  └── train.yaml           # 训练超参配置
```

## 执行计划

| 步骤 | GPU | 内容 | 预估时间 |
|------|-----|------|---------|
| 1. 数据准备 | 不需要 | 筛选 500 类目 + LLM 生成描述 | 30 分钟 |
| 2. 编写训练代码 | 不需要 | ITC dataset, LoRA config, trainer + 单元测试 | 2-3 小时 |
| 3. 首次训练 | 需要 | 2万子集, 3 epochs | 60-90 分钟 |
| 4. 评测 | 需要 | 重新 embedding + pipeline 评测 | 20 分钟 |
| 5. 对比实验 | 需要 | rank=4/8/16 | 3-4 小时 |
| 6. 收尾 | 不需要 | 更新文档, commit + tag v2.0 | 30 分钟 |

## V1 已知问题（V2 一并修复）

- `get_multimodal_embeddings_batch` 忽略了 text 参数（实际只用了图片）
- 伪层级（L1/L2）是随机分组，非语义层级 — 训练可用，报告时需注明

## 依赖

- peft==0.14.0（锁定版本，兼容 transformers 5.x）
- openai SDK（调用 MiniMax API）
- 用户提供：MiniMax API key

## 风险

| 风险 | 应对 |
|------|------|
| Q-Former LoRA 目标模块名称可能因 transformers 版本不同 | GPU 服务器上先 `model.qformer.named_modules()` 检查 |
| fp16 + LoRA dtype 不匹配 | LoRA adapter 默认 fp32，需显式处理 dtype |
| 500 类目子集改变了任务难度 | V1/V2 对比必须用同一子集 |
| LLM 生成描述质量不稳定 | 训练前抽样检查，发现问题手动修正 |
| loss 不收敛 | epoch 1 后手动检查 checkpoint，必要时调 lr |
