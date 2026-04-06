# 开发日志 — 多模态商品类目识别

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
