# 文渊 2.0 实施计划（v3 最终版）

> 基于多轮讨论和调研的完整共识。经用户确认后执行。

## 背景

文渊项目面向零基础文科博士研究者。本次 v2 集中解决 P0 bug + P1 体验优化 + 结构性改进，**一条线走完不返工**。

Karpathy 式知识库升级留作 v3 vision。

---

## 第零步：仓库结构重组

### 当前状态
`aiforhumanities/` 根目录即 vault，`.git` 指向 `cyq1017/wenyuan`。

### 目标状态
```
aiforhumanities/              ← 本地开发工作区（不推 GitHub）
├── wenyuan/                  ← vault，有自己的 .git → cyq1017/wenyuan
│   ├── 00_收件箱/
│   ├── 10_日记/
│   ├── 20_项目/              ← 原 20_论文/
│   ├── 30_文献/
│   ├── 40_概念/
│   ├── 50_资源/
│   ├── 90_计划/
│   ├── 99_系统/
│   ├── .claude/
│   ├── .agents/
│   ├── CLAUDE.md
│   ├── GEMINI.md
│   ├── AGENTS.md
│   ├── README.md
│   ├── LICENSE
│   └── .gitignore
├── docs/                     ← 开发记录（本地，按项目组织）
│   └── wenyuan/              ← 文渊项目开发记录
│       ├── v1-walkthrough.md ← 从 Antigravity 对话记录导出
│       ├── v2-changelog.md
│       ├── decisions.md      ← 关键设计决策记录
│       └── session-notes.md  ← Antigravity session 衔接笔记
└── README.md                 ← 工作区说明（本地）
```

### Antigravity 进度衔接

`docs/wenyuan/session-notes.md` 作为 Antigravity session 之间的记忆桥梁：
- 每次 session 结束时，将关键决策、当前进度、下一步写入此文件
- 下次新 session 开始时，指向 `docs/wenyuan/` 即可恢复上下文
- 类似 orbit 的"启动提示词"，但持久化在文件中，不丢失

### 操作步骤
1. 在 `aiforhumanities/` 下创建 `wenyuan/` 子目录
2. 将所有 vault 文件移入 `wenyuan/`
3. 在 `wenyuan/` 中重新 `git init`，设置 remote 为 `cyq1017/wenyuan`
4. 创建 `docs/wenyuan/` 目录，导入 v1 开发记录
5. 清理 `aiforhumanities/` 根目录原有的 `.git`

---

## 第一步：Skills 系统全面修复（一次性完成）

### 1.1 迁移到 `.claude/skills/` 目录

**背景调研结论：** Claude Code 已将 `commands/` 和 `skills/` 合并。`.claude/skills/xxx/SKILL.md` 同时支持 slash command 和 auto-invocation（根据用户意图自动触发）。

**操作：**
- 将 `.agents/skills/` 中的 12 个 SKILL.md 复制到 `.claude/skills/` 对应目录
- 删除 `.claude/commands/` 中的 9 个间接引用文件（不再需要）
- 保留 `.agents/skills/` 目录（Gemini/Codex 通过 AGENTS.md 引用）

### 1.2 同时完成所有 skill 改造

| Skill | 操作 | 说明 |
|-------|------|------|
| `start-my-day` | **改造** | 简洁化 + 加交互问答 + 去 emoji + 收件箱自动提取 |
| `literature-review` | 保留 | 微调路径引用（20_论文→20_项目） |
| `explore-concept` | 保留 | 微调路径引用 |
| `advance-paper` | **改造** | 论文→项目，措辞通用化 |
| `writing-assist` | 保留 | 无变化 |
| `organize-notes` | 保留 | 无变化 |
| `archive` | 保留 | 微调路径引用 |
| `ask` | 保留 | 无变化 |
| `faculty-search` | **改造→academic-feed** | 扩展为学术动态推送，读取偏好设置 |
| `obsidian-markdown` | 保留 | 参考文档，不改 |
| `obsidian-bases` | 保留 | 参考文档，不改 |
| `json-canvas` | 保留 | 参考文档，不改 |
| **research** | **新增** | 学术综述/主题调研，orbits 的双阶段 agent 模式 |
| **brainstorm** | **新增** | 头脑风暴，模糊想法 → 清晰 |

### 1.3 start-my-day 改造细节

参照 orbit 版本重写，学术化：

```
Step 1: 静默收集（获取日期、昨日日记、活跃项目、收件箱、偏好设置）
Step 2: 交互问答（Q1 今日重点、Q2 新想法→收件箱、Q3 困难）
Step 3: 生成日记（基于模板，填充待办+项目状态）
Step 4: 处理 Q2 新想法（自动创建收件箱条目）
Step 5: 结构化汇报
```

输出风格：orbit 的简洁结构化，不要长篇温馨问候。

### 1.4 新增 research skill

```
/research <主题>

Step 1: 理解用户要调研的主题
Step 2: 生成调研计划（涉及哪些概念、哪些文献方向、研究框架）
Step 3: 用户确认计划
Step 4: 执行调研（利用 AI 知识 + vault 中已有笔记）
Step 5: 生成研究综述 → 30_文献/ 目录
Step 6: 自动创建/更新相关概念卡片 → 40_概念/
```

### 1.5 新增 brainstorm skill

```
/brainstorm

Step 1: 问用户想聊什么（选题方向、研究角度、论证困难等）
Step 2: 发散式提问，帮助梳理
Step 3: 整理讨论成果（关键点、可能方向、推荐下一步）
Step 4: 询问是否存为收件箱条目或直接 /kickoff 立项
```

### 1.6 academic-feed skill（原 faculty-search）

```
/academic-feed

读取 99_系统/偏好设置.md 中的关注配置
搜索：教职招聘、新论文、期刊 CFP、学术会议
整理结果，按用户偏好过滤
```

同时创建 [NEW] `99_系统/偏好设置.md`：

```yaml
---
type: config
---
# 偏好设置

## 研究领域
- 主领域:
- 关注方向:

## 学术动态关注
- 教职招聘: true
- 新论文推送: true
- 期刊征稿/CFP: true
- 学术会议: true

## AI 助手
- 默认引用格式: GB/T 7714
```

---

## 第二步：模板与配置修复

### 2.1 日记模板去 emoji + 加当日总结

##### [MODIFY] `99_系统/模板/日记模板.md`

```markdown
---
type: daily
date: "{{date:YYYY-MM-DD}}"
---
# {{date:YYYY-MM-DD}} 学术日志

## 今日计划
> [由 AI 在 /start-my-day 中填充]

## 阅读与思考
-

## 写作进展
-

## 灵感与思考
-

## 学术动态
> [由 AI 在 /start-my-day 中填充]

## 相关项目
-

## 当日总结

**完成:**
-

**未完成:**
-

**明日重点:**
1.
2.
3.
```

### 2.2 项目模板更名

##### [RENAME] `论文项目模板.md` → `项目模板.md`
- "论文项目" → "项目"
- C.A.P. 结构保留，说明适用于论文、课题、研究计划等

### 2.3 修复 settings.json 权限

##### [MODIFY] `.claude/settings.json`

```json
{
  "permissions": {
    "allow": [
      "Read(*)",
      "Write(00_收件箱/**)",
      "Write(10_日记/**)",
      "Write(20_项目/**)",
      "Write(30_文献/**)",
      "Write(40_概念/**)",
      "Write(50_资源/**)",
      "Write(90_计划/**)",
      "Write(99_系统/归档/**)",
      "Write(99_系统/模板/**)",
      "Write(99_系统/偏好设置.md)"
    ]
  }
}
```

### 2.4 区分 AI 工具配置文件

##### [MODIFY] `CLAUDE.md`
- Claude Code 专属：保留 slash 命令列表，说明 `/命令名` 触发

##### [MODIFY] `GEMINI.md`
- 移除 slash 命令引用，改为自然语言触发说明
- 引用 `.agents/skills/` 目录

##### [MODIFY] `AGENTS.md`
- 通用版（Codex/Kimi 等），自然语言触发

---

## 第三步：结构性调整

### 3.1 "20_论文" 改名 "20_项目"

##### [RENAME] `20_论文/` → `20_项目/`

同步更新所有引用了 `20_论文` 的文件（CLAUDE.md、GEMINI.md、AGENTS.md、settings.json、所有 SKILL.md、模板）。

### 3.2 收件箱工作流明确化

保持简洁定位：**灵感和想法的捕捉站**。

流转路径：
```
用户随时记录 ──────→ 00_收件箱/（手动）
start-my-day Q2 ──→ 00_收件箱/（AI 自动创建）
                        │
                ┌───────┼────────┐
                ↓       ↓        ↓
            /kickoff  /read   /explore
            想法→项目  →文献    →概念
                │       │        │
                ↓       ↓        ↓
            20_项目/  30_文献/  40_概念/

处理后标记 status: processed → 定期 /archive 归档
```

---

## 第四步：README 重构 + 教程体系

### 4.1 README 重构

##### [MODIFY] `README.md`

**核心思路转变**：不教人装 Node.js，而是——

> 先学会 Obsidian 和 Claude Code（附学习资源），再一键装文渊。
> 如果你已经会用 Claude Code，直接让它帮你装文渊。

**结构大纲：**

```markdown
# 文渊 (WenYuan)
> 专为文科研究者设计的 AI 学术助手

## 能做什么？（表格，微调措辞）

## 开始之前

### 1. 学会使用 Obsidian
- [Obsidian 官方中文文档](链接)
- [PKMer 社区教程](https://pkmer.cn)
- B站搜索: "Obsidian 保姆级教程"

### 2. 学会使用 Claude Code
- [Claude Code 官方文档](链接)
- [Claude Code 入门教程](链接)
- 费用: Pro $20/月 起
- 用不了 Anthropic？→ 见下方"使用其他模型"

### 3. 使用其他模型（可选）
通过 cc-switch / OpenRouter 将 Claude Code 连接到其他模型：
- Kimi、MiniMax、DeepSeek（国产）
- Gemini、GPT（国外）
- 设置方法: [cc-switch 教程链接]

## 安装文渊

### 方式一: 让 Claude Code 帮你装（推荐）
> 在 Claude Code 中说:
> "帮我从 GitHub 下载 cyq1017/wenyuan 到我的文档文件夹"

### 方式二: 网页下载 ZIP（零终端）
1. 打开 https://github.com/cyq1017/wenyuan
2. Code → Download ZIP → 解压
3. Obsidian 中打开文件夹

### 方式三: 命令行
npx degit cyq1017/wenyuan 我的文渊

## 安装完成后
→ 打开 [[99_系统/快速上手教程]]
→ 或查看 [[99_系统/文渊使用指南]]

## 其他 AI 工具（实验性）
文渊也兼容以下工具，用自然语言描述需求即可：
- [Gemini CLI](链接) | [Codex CLI](链接) | [Kimi Code](链接)

## 文件夹说明（更新: 20_项目）
## 命令速查表（标注 CLI 专用）
## 常见问题（更新）
## 设计理念（保留）
```

### 4.2 使用指南

##### [NEW] `99_系统/文渊使用指南.md`

参照 orbit 使用指南结构，学术化改编：

```markdown
# 文渊使用指南

## 每日工作流
## 核心命令速查（CLI）
## 自然语言触发词（其他工具）
## 目录结构
## 项目标准格式（C.A.P.）
## 收件箱工作流
## Skills 一览

## 进阶
### Obsidian 插件推荐（按需选装，附教程链接）
### 多端同步（几种方案对比，用户自选）
### 个性化定制（修改文件夹、自建 skill、模板）
```

### 4.3 手把手 Tutorial

##### [NEW] `99_系统/快速上手教程.md`

从一个收件箱想法出发，串联所有功能：

```
场景：纸质课堂笔记能不能电子化归档到文渊？

Step 1: /start-my-day → 认识日记系统
Step 2: 把想法放进收件箱 → 认识收件箱
Step 3: /kickoff → 想法变项目 → 认识 C.A.P.
Step 4: OCR → /organize → 认识笔记整理
Step 5: /read → 精读笔记中引用的论文 → 认识文献系统
Step 6: /explore → 探索不熟的概念 → 认识概念卡片
Step 7: /write → 润色一段论述 → 认识写作辅助
Step 8: 回到日记总结 → 认识完整生命周期
```

---

## 第五步：清理

### 5.1 删除空壳配置目录
- 移除 `.gemini/`、`.codex/`、`.kimi/` 空目录
- 功能通过 GEMINI.md / AGENTS.md 实现，不需要空目录

### 5.2 Obsidian 技术 skills 归类
- `obsidian-markdown`、`obsidian-bases`、`json-canvas` 从使用指南核心命令中移除
- 保留在 `.agents/skills/` 作为 AI 参考文档

---

## 完整文件变更汇总

| 操作 | 文件 | 说明 |
|------|------|------|
| 重组 | 全部 vault 文件移入 `wenyuan/` | 仓库结构调整 |
| 迁移 | `.agents/skills/*` → `.claude/skills/*` | Skills 迁移到新标准目录 |
| 删除 | `.claude/commands/*`（9个） | 不再需要间接引用 |
| 改造 | `start-my-day/SKILL.md` | 简洁化+交互+去emoji |
| 改造 | `advance-paper/SKILL.md` | 论文→项目 |
| 改造+重命名 | `faculty-search/` → `academic-feed/` | 学术动态推送 |
| 新增 | `research/SKILL.md` | 学术综述调研 |
| 新增 | `brainstorm/SKILL.md` | 头脑风暴 |
| 改造 | `日记模板.md` | 去emoji+加当日总结 |
| 重命名 | `论文项目模板.md` → `项目模板.md` | 通用化 |
| 改造 | `.claude/settings.json` | 加模板和偏好写权限 |
| 改造 | `CLAUDE.md` | Claude Code 专属配置 |
| 改造 | `GEMINI.md` | 自然语言触发+引用skills |
| 改造 | `AGENTS.md` | 通用自然语言触发 |
| 重命名 | `20_论文/` → `20_项目/` | 通用化 |
| 重写 | `README.md` | 先学工具再装文渊 |
| 新增 | `99_系统/文渊使用指南.md` | 全功能指南 |
| 新增 | `99_系统/快速上手教程.md` | 手把手tutorial |
| 新增 | `99_系统/偏好设置.md` | 用户自定义 |
| 删除 | `.gemini/` `.codex/` `.kimi/` | 清理空壳目录 |
| 新增 | `docs/`（本地） | 开发记录 |

---

## Verification Plan

### 自动验证
1. `.claude/skills/` 中每个 SKILL.md 包含完整指令
2. `.claude/commands/` 目录已清空或删除
3. `settings.json` 权限正确
4. 所有文件中 `20_论文` 已替换为 `20_项目`
5. 日记模板无 emoji
6. `wenyuan/` 目录的 git remote 指向 `cyq1017/wenyuan`

### 手动验证
- 在 `wenyuan/` 中启动 Claude Code，测试 `/start-my-day`、`/read`、`/explore`、`/research`、`/brainstorm`
- 确认 slash commands 自动识别（`.claude/skills/` 生效）
- README 的 GitHub 下载链接正确
