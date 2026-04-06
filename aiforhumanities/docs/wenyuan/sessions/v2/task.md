# 文渊 2.0 执行任务

## 第零步：仓库结构重组
- [x] 创建 `wenyuan/` 子目录
- [x] 将 vault 文件移入 `wenyuan/`
- [x] 20_论文 改名 20_项目
- [x] 在 `wenyuan/` 中初始化 git，设置 remote
- [x] 创建 `docs/wenyuan/` 目录
- [x] 导入 v1 开发记录到 `docs/wenyuan/`
- [x] 清理根目录原有 `.git`

## 第一步：Skills 系统全面修复
- [x] 在 `wenyuan/.claude/skills/` 下创建所有 skill 目录
- [x] 迁移+改造 start-my-day（简洁化+交互+去emoji）
- [x] 迁移 literature-review（微调路径）
- [x] 迁移 explore-concept（微调路径）
- [x] 迁移+改造 advance-paper（论文→项目）
- [x] 迁移 writing-assist
- [x] 迁移 organize-notes
- [x] 迁移+改造 archive（路径+去emoji）
- [x] 迁移 ask
- [x] 改造 faculty-search → academic-feed
- [x] 迁移 obsidian-markdown / obsidian-bases / json-canvas
- [x] 新增 research skill
- [x] 新增 brainstorm skill
- [x] 删除 `.claude/commands/` 旧文件
- [x] 创建 `99_系统/偏好设置.md`
- [x] 同步新 skills 到 `.agents/skills/`（Gemini/Codex 兼容）

## 第二步：模板与配置修复
- [x] 日记模板去 emoji + 加当日总结
- [x] 项目模板更名（论文→项目）
- [x] 所有模板去 emoji（概念卡片、文献精读等）
- [x] 修复 settings.json 权限
- [x] 改造 CLAUDE.md（Claude Code 专属）
- [x] 改造 GEMINI.md（自然语言触发）
- [x] 改造 AGENTS.md（通用版）

## 第三步：README 重构 + 教程体系
- [x] 重写 README.md
- [x] 新建 99_系统/文渊使用指南.md
- [x] 新建 99_系统/快速上手教程.md

## 第四步：清理
- [x] 删除 .gemini/ .codex/ .kimi/ 空目录
- [x] 创建 docs/wenyuan/session-notes.md（进度衔接）

## 第五步：验证
- [x] 检查所有文件路径引用正确（无 20_论文 残留）
- [x] 检查所有模板无 emoji
- [x] 检查 git remote 正确
- [ ] 推送到 GitHub（需用户确认）
