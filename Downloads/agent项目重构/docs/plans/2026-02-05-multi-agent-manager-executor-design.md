# 多 Agent（管理者-执行者）最小样例设计

## 背景与目标
本次更新目标是为现有车载对话系统引入“管理者-执行者”多 Agent 协作能力，以一个最小样例验证端到端链路可行性。最小样例为用户显式口令“出行准备”，由管理者 Agent 规划并并行分派给天气/路线/音乐三个执行 Agent，结果按计划顺序流式输出。

## 范围与非目标
- 范围：新增 Manager/Executor 角色与最小编排能力；复用现有 NLU/Function Call/DM。
- 非目标：完整 Agent Runtime 重构、跨会话长期记忆体系、复杂任务恢复策略。

## 触发条件
- 触发口令：用户输入包含明确高层意图“出行准备”。
- 触发时机：仲裁为 task 后优先走 Manager 路径，避免与普通 NLU 冲突。

## 架构概览
- 新增 `agent/` 模块：
  - `agent/manager.py`：生成结构化计划（JSON Schema）。
  - `agent/executors/weather.py`：天气子任务执行。
  - `agent/executors/maps.py`：路线子任务执行。
  - `agent/executors/music.py`：音乐子任务执行。
  - `agent/schema.py`：计划与子任务的 JSON Schema。
- 入口改造：`start.py` 在 task 分支中新增“高层意图检测→Manager→并行调度→流式输出”。
- Prompt 扩展：`prompts.py` 增加 Manager/Executor system prompt。

## 计划与执行模型
- Manager 通过 LLM 输出计划：
  - `plan_id`、`tasks[]`、`task_id`、`type`、`priority`、`inputs`、`expected_output`。
- Executors 仅负责各自领域的参数补齐、工具调用与结果整理。
- 计划解析失败或 Schema 校验失败时，回退到原有 NLU 逻辑。

## 数据流
1. 用户输入“出行准备”。
2. 仲裁为 task，命中高层意图。
3. Manager 生成计划（JSON）。
4. 并行触发 3 个 Executor。
5. 聚合器按计划顺序流式输出三段（天气→路线→音乐）。
6. Redis 写入 plan/task 状态与最终摘要。

## 并行与合并策略
- Executor 并行运行，结果以“计划顺序”输出，避免乱序影响用户理解。
- 每个子任务输出一段自然语言回复，最后追加一个简短汇总。

## 错误处理与回退
- 子任务失败：输出失败提示 + 简短替代建议，不中断其他任务。
- 外部工具超时：Executor 返回可读错误信息。
- Manager 计划解析失败：回退原有 NLU。
- 日志包含 `trace_id + plan_id + task_id` 便于排查。

## 测试与验收
- 单元测试：Schema 校验、Executor 参数补齐、失败回退。
- 集成测试：触发“出行准备”时三段输出完整。
- 并发测试：乱序返回时仍按计划顺序输出。
- 回退测试：计划解析失败时走原流程。

## 里程碑
- M1：Manager/Executor 结构与最小路由完成。
- M2：并行执行与流式输出完成。
- M3：测试覆盖与验收。
