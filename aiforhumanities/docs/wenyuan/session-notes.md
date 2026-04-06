# 文渊开发 Session 衔接笔记

> Antigravity session 之间的记忆桥梁。新 session 开始时读取此文件恢复上下文。

---

## 开发记录归档

| 版本 | 日期 | 对话ID | 本地归档 |
|------|------|--------|----------|
| v1 | 2026-03-31 | `4f366c0c` | `docs/wenyuan/v1-session-2026-03-31/` |
| v2 | 2026-04-05~06 | `e75ee652` | `docs/wenyuan/v2-session-2026-04-05/` |

每个归档包含该轮对话的 `implementation_plan.md`、`task.md`、`walkthrough.md`（如有）。

> **注意：** docs 目录只做本地 commit，不推送到 remote（wenyuan 仓库只推 vault 内容）。

---

## 最近 Session: 2026-04-06 (v2.3)

### 完成工作

**v2.0 (基础重构):**
- [x] 仓库重组：vault 移入 `wenyuan/` 子目录，git 指向 `cyq1017/wenyuan`
- [x] Skills 全面迁移：14 个 skill 到 `.claude/skills/` + `.agents/skills/`
- [x] 新增 `/research`、`/brainstorm`、`/academic-feed`
- [x] 所有路径引用更新：20_论文 → 20_项目
- [x] 所有模板去 emoji
- [x] settings.json 权限修复
- [x] CLAUDE.md / GEMINI.md / AGENTS.md 配置分工

**v2.1 (文档完善):**
- [x] README 增加文渊命名来源和 LLM 知识库愿景
- [x] start-my-day 集成学术动态推送
- [x] 新建 Obsidian 新手指南（含 Terminal、Web Clipper 插件配置）
- [x] 新建 Claude Code 新手指南（含 cc-switch、国内订阅、防封号）
- [x] 重写快速上手教程（具体场景：纸质笔记 OCR 归档）
- [x] 使用指南扩展个性化定制教程

**v2.2 (end-my-day):**
- [x] 新增 `/end-my-day` 每日复盘（参考本地 orbit，含对话归档）
- [x] 所有配置文件同步更新

**v2.3 (用户反馈修订):**
- [x] README 愿景改写为更含蓄的表达（文渊阁典故）
- [x] 不限定文科，改为"主要面向人文社科，任何学科可定制"
- [x] start-my-day 标注推送学术动态
- [x] academic-feed 标注为已集成到 start-my-day
- [x] "开始之前"改为内部教程链接跳转，去掉费用冗余
- [x] Claude Code 指南：cc-switch 改正确链接、新增国内中转平台、鼓励问 AI
- [x] Obsidian 指南：标明必装插件、整合外部教程链接
- [x] Tutorial 重写为"想法→计划→项目→执行→积累"流程
- [x] Tutorial 加入 start-my-day / end-my-day 介绍
- [x] Tutorial 加入自建 skill（OCR 归档）作为个性化定制示例
- [x] FAQ 加入手机同步和个性化定制入口

### 关键决策记录
- 主推 Claude Code + 终端（Obsidian Terminal 插件内使用）
- Obsidian + Claude Code 是一个整体，不是分开的工具
- 替代模型方案用 OpenRouter/cc-switch，完全合规
- docs/ 只做本地 commit，不推 remote
- remote 只推 wenyuan vault 本身的内容
- Karpathy 式知识库升级留 v3

### 待办 / 下一步
- [ ] 用户在 Obsidian 中实际测试完整工作流
- [ ] 验证 Terminal 插件内 Claude Code 对 skills 的识别
- [ ] v3 愿景规划：AI 自动编译知识库、健康检查、自动关联

### 项目状态
- **仓库位置**: `/Users/caoyuqi/aiforhumanities/wenyuan/`
- **GitHub**: https://github.com/cyq1017/wenyuan
- **当前版本**: v2.3（已推送 remote）
- **最后 commit (remote)**: bc73f6c
