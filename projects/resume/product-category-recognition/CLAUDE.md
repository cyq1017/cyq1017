# BLIP2 多模态商品类目识别

## 项目概述
基于BLIP2的商品类目识别系统，三阶段推理框架（召回→排序→重排）。简历项目#2。

## 技术栈
- Python 3.10+, PyTorch, Transformers, PEFT(LoRA)
- BLIP2 (Salesforce/blip2-opt-2.7b), Q-Former, FAISS
- pytest (测试), matplotlib/seaborn (可视化)

## 项目结构
```
src/model/blip2_encoder.py    # BLIP2模型封装 (适配transformers 5.x)
src/offline/                   # 离线: embedding生成 + FAISS索引
src/online/                    # 在线: 召回→排序→重排流水线
src/evaluation/metrics.py      # TopK准确率 + 混淆矩阵
scripts/                       # 入口脚本 (run_offline, evaluate, demo)
tests/                         # 单元测试 (23个)
config/default.yaml            # 超参配置
data/                          # 数据 + 合成数据生成器
docs/design.md                 # 架构设计文档
```

## 快速命令
```bash
make data       # 生成合成数据
make offline    # 离线embedding + FAISS索引 (需GPU)
make evaluate   # 在线推理 + 评测
make all        # 全流程
pytest tests/   # 运行测试 (不需GPU)
```

## 开发规范
- 遵循 superpowers 流程: brainstorming → plan → TDD → verify → review
- 每个版本 git tag 保留 (v1.0, v2.0)
- 测试覆盖率 80%+
- 代码变更后运行 `pytest tests/ -v`

## 目标指标
- V1 (zero-shot): 叶子TOP1 72%, 二级 82.4%, 一级 88.5%
- V2 (LoRA): 叶子+3.2%, 二级+4.0%, 一级+5.6%

## 关键技术备忘
- transformers 5.x 移除了 `get_text_features` / `get_qformer_features`
- Q-Former cross-attention 必须有 image input (不支持 text-only)
- text embedding 用 LM word_embeddings, image 用 Q-Former + language_projection
- 需要 `from __future__ import annotations` 兼容 Python 3.9
- GPU服务器需设 `export HF_ENDPOINT=https://hf-mirror.com`

## 当前状态
- V1 已完成并 tag v1.0 (leaf_top1=24.6% on Products-10K 9691 SKU)
- 下一步: V2 LoRA 微调 Q-Former
- 详见 docs/design.md, docs/devlog.md

## Orbit 交接协议
- 开始工作时：
  `obsidian vault=claudecode_workspace read file="商品类目识别"`
- 结束工作时（≤500 token）：
  `obsidian vault=claudecode_workspace append file="商品类目识别" content="交接记录"`
- 只读状态：
  `obsidian vault=claudecode_workspace property:read name=status file="商品类目识别"`
- 交接记录含：完成/放弃的方案/技术下一步
- 注意：用 file= 不用 path=

## 进度同步
- 阶段性完成时更新 orbit 笔记进展 + docs/devlog.md
- 结束工作时更新 orbit 笔记 + commit

## SOP 参考（需要时读取，不要全部加载）
- 开发流程: /Users/caoyuqi/claudecode_workspace/orbit/30_研究/AI工程/开发流程SOP.md
- 进度保护: /Users/caoyuqi/claudecode_workspace/orbit/30_研究/AI工程/Claude_Code进度管理/进度保护SOP.md

## 项目文档
- docs/design.md — 架构设计（三阶段推理、数据方案、技术备忘）
- docs/project_plan.md — 项目计划（阶段目标、技术决策）
- docs/devlog.md — 开发日志（每次会话的工作记录和踩坑）
- docs/dialogs/ — 会话原始对话备份（jsonl，hook 自动生成）
- docs/handoff/ — Context Reset 时的交接文档
