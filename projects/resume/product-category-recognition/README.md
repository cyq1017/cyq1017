# 基于BLIP2多模态大模型的商品类目识别

## 项目简介

针对电商场景，利用BLIP2的Zero-shot能力实现商品信息对齐和类目识别。通过三阶段推理框架（召回→排序→重排）实现高精度的商品自动分类。

## 系统架构

```
离线阶段:
  锚商品(文本+图片) → BLIP2 Q-Former → 多模态Embedding → 存储
  类目文本描述      → BLIP2          → 文本Embedding   → 存储

在线阶段:
  新商品 → Embedding提取
       ↓
  [召回] 多模态Embedding vs 锚商品Embedding → TopN锚商品 → 加权候选类目
       ↓
  [排序] 图片Embedding vs 类目文本Embedding → TopN叶子类目
       ↓
  [重排] 融合召回+排序分数 → Top1类目预测
```

## 项目指标

| 版本 | 叶子类目TOP1 | 二级类目TOP1 | 一级类目TOP1 |
|------|-------------|-------------|-------------|
| V1   | 72.0%       | 82.4%       | 88.5%       |
| V2 (LoRA) | +3.2%  | +4.0%       | +5.6%       |

## 快速开始

### 环境安装

```bash
# 方式1: 脚本安装 (推荐)
bash setup.sh
conda activate blip2-product

# 方式2: Docker
docker build -t blip2-product .
docker run --gpus all blip2-product
```

### 一键运行完整流程

```bash
make all    # 数据生成 → 离线Embedding → 评测
```

### 分步运行

```bash
make data      # 生成合成数据和类目体系
make offline   # 离线阶段: 生成Embedding + 建索引
make evaluate  # 在线推理 + 评测指标
make demo      # 单商品演示
```

## 项目结构

```
├── config/default.yaml         # 超参配置
├── data/
│   ├── categories.json         # 类目体系
│   └── generate_data.py        # 合成数据生成器
├── src/
│   ├── model/blip2_encoder.py  # BLIP2模型封装
│   ├── offline/
│   │   ├── embedding_generator.py  # Embedding生成
│   │   └── index_builder.py        # FAISS索引构建
│   ├── online/
│   │   ├── recall.py           # 召回模块
│   │   ├── ranking.py          # 排序模块
│   │   ├── rerank.py           # 重排模块
│   │   └── pipeline.py         # 三阶段流水线
│   └── evaluation/metrics.py   # 评测指标
├── scripts/
│   ├── run_offline.py          # 离线阶段入口
│   ├── run_online.py           # 在线推理入口
│   ├── evaluate.py             # 评测脚本
│   └── demo.py                 # 演示脚本
├── Dockerfile                  # Docker部署
├── Makefile                    # 一键运行
└── setup.sh                    # 环境安装
```

## 技术栈

- **模型**: Salesforce/blip2-opt-2.7b
- **向量检索**: FAISS
- **深度学习框架**: PyTorch + Transformers
- **V2微调**: LoRA (PEFT)
