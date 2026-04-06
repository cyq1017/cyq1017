# 文渊 (WenYuan) — 完整构建记录

## 项目概述

为文科博士研究者创建了 AI 驱动的学术生产力系统「文渊」，参考 [OrbitOS](https://github.com/MarsWang42/OrbitOS) 架构，针对人文学科场景完全重新设计。

🔗 **GitHub**：https://github.com/cyq1017/wenyuan

---

## 工作时间线

| 阶段 | 内容 |
|------|------|
| **研究** | 逆向分析 OrbitOS 仓库：目录结构、CLAUDE.md 设计、skills 工作流、模板系统 |
| **方案设计** | 创建实施方案，把 OrbitOS 的科技/产品经理场景改造为文科博士学术场景 |
| **用户确认** | 用户批准方案，并追加三个需求：①多CLI支持 ②教职搜索 ③GitHub一键部署 |
| **构建** | 创建全部 47 个文件，包括目录结构、模板、skills、commands、配置 |
| **修复** | 发现 `.agents/skills/` 不是 Claude Code 的 slash command，补建 `.claude/commands/` |
| **完善** | 大幅重写 README 为面向零技术基础文科生的详细教程 |
| **部署** | Git 初始化，创建 GitHub 仓库 `cyq1017/wenyuan`（public），推送 |

---

## 完整文件清单（47 个文件）

### 系统指令（3 个）— 多 CLI 支持
| 文件 | 用途 | 对应工具 |
|------|------|---------|
| [CLAUDE.md](file:///Users/caoyuqi/aiforwenke/CLAUDE.md) | 系统指令 | Claude Code |
| [GEMINI.md](file:///Users/caoyuqi/aiforwenke/GEMINI.md) | 系统指令 | Gemini CLI |
| [AGENTS.md](file:///Users/caoyuqi/aiforwenke/AGENTS.md) | 系统指令 | Codex CLI / Kimi Code |

### Claude Code Slash Commands（9 个）
用户在 Claude Code 中输入 `/` 即可看到这些命令：

| 命令 | 文件 | 功能 |
|------|------|------|
| `/start-my-day` | [start-my-day.md](file:///Users/caoyuqi/aiforwenke/.claude/commands/start-my-day.md) | 晨间学术规划 |
| `/read` | [read.md](file:///Users/caoyuqi/aiforwenke/.claude/commands/read.md) | 文献精读 |
| `/explore` | [explore.md](file:///Users/caoyuqi/aiforwenke/.claude/commands/explore.md) | 概念溯源 |
| `/kickoff` | [kickoff.md](file:///Users/caoyuqi/aiforwenke/.claude/commands/kickoff.md) | 论文推进 |
| `/write` | [write.md](file:///Users/caoyuqi/aiforwenke/.claude/commands/write.md) | 写作辅助 |
| `/organize` | [organize.md](file:///Users/caoyuqi/aiforwenke/.claude/commands/organize.md) | 笔记整理 |
| `/archive` | [archive.md](file:///Users/caoyuqi/aiforwenke/.claude/commands/archive.md) | 归档 |
| `/ask` | [ask.md](file:///Users/caoyuqi/aiforwenke/.claude/commands/ask.md) | 快速问答 |
| `/faculty-search` | [faculty-search.md](file:///Users/caoyuqi/aiforwenke/.claude/commands/faculty-search.md) | 教职搜索 |

### AI Skills 详细指令（12 个）
`.claude/commands/` 是入口，`.agents/skills/` 存放详细工作流步骤：

| Skill | 文件 |
|-------|------|
| 晨间规划 | [SKILL.md](file:///Users/caoyuqi/aiforwenke/.agents/skills/start-my-day/SKILL.md) |
| 文献精读 | [SKILL.md](file:///Users/caoyuqi/aiforwenke/.agents/skills/literature-review/SKILL.md) |
| 概念溯源 | [SKILL.md](file:///Users/caoyuqi/aiforwenke/.agents/skills/explore-concept/SKILL.md) |
| 论文推进 | [SKILL.md](file:///Users/caoyuqi/aiforwenke/.agents/skills/advance-paper/SKILL.md) |
| 写作辅助 | [SKILL.md](file:///Users/caoyuqi/aiforwenke/.agents/skills/writing-assist/SKILL.md) |
| 笔记整理 | [SKILL.md](file:///Users/caoyuqi/aiforwenke/.agents/skills/organize-notes/SKILL.md) |
| 归档 | [SKILL.md](file:///Users/caoyuqi/aiforwenke/.agents/skills/archive/SKILL.md) |
| 快速问答 | [SKILL.md](file:///Users/caoyuqi/aiforwenke/.agents/skills/ask/SKILL.md) |
| 教职搜索 | [SKILL.md](file:///Users/caoyuqi/aiforwenke/.agents/skills/faculty-search/SKILL.md) |
| Obsidian Markdown | [SKILL.md](file:///Users/caoyuqi/aiforwenke/.agents/skills/obsidian-markdown/SKILL.md) |
| Obsidian Bases | [SKILL.md](file:///Users/caoyuqi/aiforwenke/.agents/skills/obsidian-bases/SKILL.md) |
| JSON Canvas | [SKILL.md](file:///Users/caoyuqi/aiforwenke/.agents/skills/json-canvas/SKILL.md) |

### 模板（5 个）
| 模板 | 文件 |
|------|------|
| 日记模板 | [日记模板.md](file:///Users/caoyuqi/aiforwenke/99_系统/模板/日记模板.md) |
| 文献精读模板 | [文献精读模板.md](file:///Users/caoyuqi/aiforwenke/99_系统/模板/文献精读模板.md) |
| 概念卡片模板 | [概念卡片模板.md](file:///Users/caoyuqi/aiforwenke/99_系统/模板/概念卡片模板.md) |
| 论文项目模板 | [论文项目模板.md](file:///Users/caoyuqi/aiforwenke/99_系统/模板/论文项目模板.md) |
| 收件箱模板 | [收件箱模板.md](file:///Users/caoyuqi/aiforwenke/99_系统/模板/收件箱模板.md) |

### 其他文件
| 文件 | 用途 |
|------|------|
| [学术伙伴.md](file:///Users/caoyuqi/aiforwenke/99_系统/提示词/学术伙伴.md) | AI 角色人格设定 |
| [文献管理.base](file:///Users/caoyuqi/aiforwenke/99_系统/数据库/文献管理.base) | 文献数据库视图 |
| [论文进度.base](file:///Users/caoyuqi/aiforwenke/99_系统/数据库/论文进度.base) | 论文进度追踪视图 |
| [概念索引.base](file:///Users/caoyuqi/aiforwenke/99_系统/数据库/概念索引.base) | 概念索引视图 |
| [README.md](file:///Users/caoyuqi/aiforwenke/README.md) | 超详细中文安装教程 |
| [LICENSE](file:///Users/caoyuqi/aiforwenke/LICENSE) | MIT 开源协议 |
| [.gitignore](file:///Users/caoyuqi/aiforwenke/.gitignore) | Git 忽略规则 |
| [settings.json](file:///Users/caoyuqi/aiforwenke/.claude/settings.json) | Claude Code 权限配置 |
| 10 × `.gitkeep` | 保持空目录在 Git 中可见 |

---

## 架构说明

```
用户输入 /start-my-day
        │
        ▼
┌─────────────────────┐
│ .claude/commands/    │ ← Claude Code slash command 入口
│ start-my-day.md      │   （Gemini/Codex/Kimi 通过 AGENTS.md 直接触发）
└────────┬────────────┘
         │ 引用
         ▼
┌─────────────────────┐
│ .agents/skills/      │ ← 详细工作流步骤（所有 CLI 工具共用）
│ start-my-day/SKILL.md│
└────────┬────────────┘
         │ 使用
         ▼
┌─────────────────────┐
│ 99_系统/模板/        │ ← Obsidian 笔记模板
│ 日记模板.md          │
└────────┬────────────┘
         │ 生成
         ▼
┌─────────────────────┐
│ 10_日记/             │ ← 用户的实际笔记
│ 2025-03-31.md        │
└─────────────────────┘
```

---

## 与 OrbitOS 的关键差异

| 维度 | OrbitOS | 文渊 |
|------|---------|------|
| 目标用户 | 科技从业者 | 文科博士研究者 |
| 项目系统 | 通用项目管理 | 论文项目 (C.A.P. 结构) |
| 研究功能 | 通用研究笔记 | 文献精读 + 概念溯源 |
| 写作支持 | 无 | 学术写作辅助（润色/逻辑/用语） |
| AI 风格 | 效率工具 | 温暖的学术伙伴（鼓励 + 苏格拉底提问） |
| 职业功能 | 无 | 教职搜索 |
| CLI 支持 | Claude + Gemini | Claude + Gemini + Codex + Kimi |
| 引用格式 | 无 | GB/T 7714 默认 |
| README | 英文简洁版 | 面向零技术基础的中文详细教程 |

---

## 后续可优化方向

1. **实际测试**：打开 Claude Code 在 vault 中测试每个 slash command
2. **Zotero 集成**：如果她使用 Zotero 管理文献，可增加引入功能
3. **专业方向定制**：了解她的具体专业后，可定制概念卡片和文献模板
4. **示例数据**：在 vault 中预置一些示例笔记，让她理解每种笔记长什么样
5. **移动端方案**：配置 iCloud Sync 让手机上也能用
