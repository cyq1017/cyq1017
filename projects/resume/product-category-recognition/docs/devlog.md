# 开发日志 — 多模态商品类目识别

## 2026-04-08

**完成:**
- V2 LoRA 全流程完成（设计→编码→训练→评测）
- 设计: spec + plan 经 review 修订完成
- 编码: Task 1-7 完成，17 个新测试（40 总计，38 本地通过）
- 数据: 500 类目子集，13680 train / 6190 test，GLM API 生成 500 条类目描述
- 训练: 3 epochs, ITC loss 2.75→2.63→2.59，LoRA rank=8，可训练参数 724,992 (0.02%)
- V2 评测结果（500 类目子集）:
  - leaf_top1: 61.7% → 68.1% (+6.4%)
  - l2_top1: 65.2% → 70.5% (+5.3%)
  - l1_top1: 84.1% → 86.9% (+2.8%)
- Checkpoint 已回传本地，Kaggle token 已清理

**阻塞/待解决:**
- Task 10 rank 对比实验（rank=4/16）待下次 GPU 会话

**踩坑记录:**
- Spec review 发现 ITC 训练 text path 和 V1 推理不一致 — CRITICAL，已修复
- Plan review 发现 LoRA 应用范围过大（全模型而非仅 Q-Former）— HIGH，已修复
- MiniMax API key 失败（401），改用 GLM API（glm-4-flash）成功
- run_offline.py 期望 title/description 字段，V2 CSV 没有，写独立 run_v2_eval.py
- heredoc 写入 f-string 引号被吃掉，sed 修复

---

## 2026-04-07

**完成:**
- 项目规范化: CLAUDE.md checklist 对照补全，Orbit 交接协议切回 CLI
- 全局 rules 生效: execution-discipline.md, orbit-handoff.md
- Memory 清理: 行为规则类 feedback 迁移到全局 rules，删除冗余条目
- .gitignore 补充 docs/dialogs/*.jsonl
- 错题本首次建立: QA-商品类目识别(4条), AI错误-商品类目识别(6条)

**阻塞/待解决:**
- GPU服务器上的 Kaggle token 下次开机需清理: `rm /root/.kaggle/kaggle.json`
- V2 LoRA 微调待下次会话启动

**踩坑记录:**
- memory vs CLAUDE.md/rules 职责要区分清楚
- Obsidian CLI installer 需更新才能用（已更新）

---

## 2026-04-06

**完成:**
- Products-10K 数据集下载 (19GB) + 解压 + 适配器
- 真实数据评测完成: leaf_top1=24.6%, l2_top1=25.4%, l1_top1=31.6%
- V1 commit + tag v1.0 (3264行, 40文件)
- 项目规范化: CLAUDE.md, docs/, orbit笔记更新

**阻塞/待解决:**
- V2 LoRA微调待开始

**踩坑记录:**
- Products-10K 解压后图片在 `train/train/` 嵌套目录下
- train.csv 有额外的 `group` 列
- 9691 SKU zero-shot top1=24.6%，距目标72%差距大，需LoRA微调

---

## 2026-04-03

**完成:**
- V1 全部源码实现 + 适配 transformers 5.x API
- GPU服务器 (RTX 5090 32GB) 离线+评测全流程跑通
- 23个单元测试全部通过
- 配置 Stop hook + PreCompact hook 防丢失
- 数据集调研: 推荐 Products-10K

**阻塞/待解决:**
- 合成数据评测指标低 (leaf_top1=3.5%), 需真实数据替换

**踩坑记录:**
- transformers 5.x 移除了 `get_text_features` / `get_qformer_features`
- Q-Former cross-attention 必须有 encoder_hidden_states (不支持text-only)
- 服务器上 CSV image_path 是Mac本地绝对路径, 需在服务器重新生成数据
- `from __future__ import annotations` 必须加, 否则 Python 3.9 报 TypeError

---

## 2026-03-31

**完成:**
- 项目初始化, 从文档重建V1代码框架
- Phase 0-6: 骨架 → 数据 → BLIP2封装 → 离线 → 在线 → 评测 → 脚本
- 合成数据生成: 22000条商品, 550叶子类目
- 1463行代码完成

**踩坑记录:**
- 代码完全丢失, 只有文档可参考
