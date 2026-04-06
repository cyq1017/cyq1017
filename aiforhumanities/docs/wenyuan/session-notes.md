# 文渊开发 Session 衔接笔记

> Antigravity session 之间的记忆桥梁。新 session 开始时读取此文件恢复上下文。

---

## 开发记录归档

| 版本 | 日期 | 对话ID | 本地归档 |
|------|------|--------|----------|
| v1 | 2026-03-31 | `4f366c0c` | `docs/wenyuan/v1-session-2026-03-31/` |
| v2 | 2026-04-05~06 | `e75ee652` | `docs/wenyuan/v2-session-2026-04-05/` |

> **注意：** docs 目录只做本地 commit，不推送到 remote。

---

## 最近 Session: 2026-04-06 (v2.4 执行中)

### 当前状态：v2.3 Review Notes 执行中

Review Notes 位于：`docs/wenyuan/v2.3-review-notes.md`

| 编号 | 文件 | 状态 | commit |
|------|------|------|--------|
| R-1~R-6 | README.md + AGENTS.md + GEMINI.md | ✅ 已完成 | `eff06b7` |
| C-1~C-5 | Claude Code 新手指南 | ✅ 已完成 | `14eab57` |
| O-1~O-5 | Obsidian 新手指南 | ✅ 已完成 | `b5e4b71` |
| G-1~G-5 | 文渊使用指南 | ⏳ 待执行 |  |
| T-1~T-2 | 快速上手教程 | ⏳ 待执行 |  |

### 待执行要点（G 和 T）

**G-1**: start-my-day 描述补充（交互问答 + 推送 + 生成日记），skill 列表补全
**G-2**: 日记模板改为四区域（阅读与思考、项目进展、灵感与想法、当日总结）
**G-3**: 50_资源/ 加入学术动态 digest 机制，偏好设置加"信息订阅"区域，tutorial 中引导用户设定
**G-4**: iCloud 同步路径说明
**G-5**: 个性化定制加"先跟 Claude Code 商量"的建议
**T-1**: 检查文渊依赖的 Obsidian 插件（Canvas、Templater？），教用户装插件
**T-2**: 空 vault 先写收件箱想法，给出"数字人文 distant reading"具体示例，演示流程链

### 已完成的改动汇总

**README (R-1~R-6):**
- 愿景改为"文以载道，渊博知识" + "越用越有价值"说明
- 去掉"主要面向人文社科"，改为"为学术研究设计，自由调整"
- 所有 `[[]]` wikilinks 加上 GitHub 相对路径链接
- 命令行方式前加终端说明
- 其他 AI 工具：Codex CLI 提前 + 增加 Kimi Code
- FAQ 措辞调整

**AGENTS.md + GEMINI.md (R-6):**
- 去掉"文科研究者"限定
- "必须使用中文"改为"默认使用中文交流"
- 语气对齐 CLAUDE.md（"像一位经验丰富的学术前辈"）

**Claude Code 指南 (C-1~C-5):**
- 开头逻辑修复：安装前问 ChatGPT，安装后问 Claude Code
- 方案排序：Anthropic 第一 → cc-switch 第二 → 中转平台第三
- 国内 Token 订阅平台链接（Kimi/DeepSeek/智谱/百炼/MiniMax）
- 中转平台设置方法改为具体终端命令
- 学习资源加 B站/知乎/小红书/公众号搜索关键词
- FAQ "不确定怎么操作"逻辑修复

**Obsidian 指南 (O-1~O-5):**
- Frontmatter 详细解释 + 字段说明
- 社区 Skills 介绍（obsidian-cli, defuddle, Axton 三件套）+ 安装命令
- iCloud 同步路径
- 笔记迁移加 `/organize` 归类
- 链接加 GitHub 路径

### 项目状态
- **仓库位置**: `/Users/caoyuqi/aiforhumanities/wenyuan/`
- **GitHub**: https://github.com/cyq1017/wenyuan
- **当前本地最新 commit**: `b5e4b71`（还未 push）
- **remote 最新**: `bc73f6c`（v2.3）
- **Review Notes**: `docs/wenyuan/v2.3-review-notes.md`
