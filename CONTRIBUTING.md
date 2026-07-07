# Contributing

## 推荐工作流

1. 从 `main` 分支创建功能分支：
   - `feature/hmi-panel-*`
   - `feature/headless-validation-*`
   - `feature/llm-wrapper-*`
2. 小步提交，提交信息说明改动目的。
3. 提 PR 前至少运行一次相关 smoke test。
4. PR 描述中写明：
   - 改动模块
   - 运行方式
   - 验证结果
   - 是否影响安全验证逻辑

## 不应提交的内容

- `__pycache__/`
- `*.pyc`
- BlueSky 运行输出：`bluesky_project/output/`
- BlueSky 缓存：`bluesky_project/data/cache/`
- 大规模 headless 日志：`headless_validation/headless_dynamic_logs/`
- 模型权重、私钥、token、API key

## 安全边界

涉及冲突解脱策略时，必须保持：

- LLM 不直接下发未经验证的 BlueSky 命令；
- 所有动作 ID 必须来自候选动作集合；
- 安全验证失败时不得展示为可执行指令；
- GUI fallback 不能替代 headless safety evidence。
