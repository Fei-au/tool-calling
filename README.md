# Tool Calling Weather Demo

这个仓库展示了两种方式实现“网上查询天气”的 tool calling 学习示例，两者都是**真正的 agent 循环**（由模型自己决定是否调用工具、调用几次，直到给出最终回答）：

1. 直接使用 OpenAI 的 tool calling API，手写 agent 循环
2. 使用 LangGraph 编排同一个 agent 循环

## 目录说明

- [main.py](main.py): 直接调用 OpenAI tool calling 的示例，以及 LangGraph 版本
- [weather_tool.py](weather_tool.py): 联网查询天气的工具函数，使用 Open-Meteo API
- [requirements.txt](requirements.txt): Python 依赖
- [.env.example](.env.example): 环境变量示例

## 安装依赖

```bash
cd /Users/yafei/code/tool-calling
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

然后把你的 OpenAI API Key 写入 .env：

```bash
OPENAI_API_KEY=your-openai-api-key-here
```

## 运行示例

### 1) 直接使用 OpenAI tool calling

```bash
python main.py --mode direct --message "今天上海的天气怎么样"
```

### 2) 使用 LangGraph 编排

```bash
python main.py --mode langgraph --message "今天上海的天气怎么样"
```

## 支持的查询范围

当前工具支持这些时间范围：

- 今天
- 7天后
- 14天后
- 1个月（30天）

例如：

```bash
python main.py --mode direct --message "7天后的北京天气怎么样"
python main.py --mode langgraph --message "14天后的深圳天气怎么样"
python main.py --mode direct --message "一个月后的杭州天气怎么样"
```

## 多轮工具调用（agent 循环的核心）

一次 `lookup_weather` 通常就能拿到答案，所以大多数问题只跑一轮。真正有价值的**多轮**调用（`assistant → tool → assistant → tool → assistant`），关键**不是定义更多工具**，而是让**下一轮查什么，取决于模型对上一轮结果的推理判断**——参数是模型「想」出来的，不是上个工具直接转手给它的。

为此仓库提供了两个配合使用的工具，形成「先看概览 → 再钻取细节」的 agentic drill-down：

- `lookup_weather(location, timeframe)`：**日报概览**（每天的高低温、降雨概率）
- `get_hourly_forecast(location, date)`：**某一天的逐小时明细**（温度、体感、降雨概率、紫外线）

示例问题：

```bash
python main.py --mode direct --message "我想这周找一天去上海户外骑行，帮我挑降雨最少的那天，再告诉我那天几点出门最舒服"
```

实际执行过程：

```
第1轮  lookup_weather("上海","7days")          → tool: 整周日报（含各天降雨概率）
        ⇩ 模型比较 7 天，推理出 7/14 降雨最少        ← 这一步是 LLM 在「想」
第2轮  get_hourly_forecast("上海","2026-07-14") → tool: 那一天的逐小时数据
        ⇩ 模型在小时数据里挑出最舒适的时段
第3轮  assistant → 最终回答（哪天最好 + 几点出门）
```

关键点：第 2 轮的 `date="2026-07-14"` **不是上一个工具返回的**，而是模型看完整周日报后**自己推理选出来的**。这也无法压缩成一轮——不看完整周就不知道该钻取哪天。这才是真正的 agent（会做决策），而不只是把参数从 A 转手给 B。

对比一下两种“多次调用”：

- **多轮（串行，需要推理）**：下一轮的参数依赖模型对上一轮结果的判断（如上例），必须一轮一轮来。
- **一轮内并行（相互独立）**：工具间没有依赖（如“对比北京和成都”），模型会在**同一轮**里一次性发出两个 `lookup_weather` 调用。

两种模式（`direct` 和 `langgraph`）都会走完整的多轮循环：

```bash
python main.py --mode langgraph --message "帮我看未来一周北京哪天最适合户外活动，再告诉我那天几点出门最舒服"
```

## 学习计划（Roadmap）

由浅入深，一步步从"会调一个工具"走到"能自己反复思考、多轮执行的 agent"。

### 阶段 1 · 吃透现有代码的执行过程
- 逐行读懂 [main.py](main.py) 的 agent 循环（`for _ in range(max_steps)`）：模型如何决定「调工具 or 给最终答案」，`tool_calls` 如何回填进 `messages`。
- 打印每一轮的消息流，亲眼看到 `system → user → assistant(tool_calls) → tool → assistant`。
- 弄清 function tool 的三件事：JSON schema 定义、模型只吐参数、真正执行的是自己的代码。

### 阶段 2 · 精读 Anthropic 的 tool use 文档
- 官方文档：Anthropic「Tool use」+「Building agents with the Claude Agent SDK」。
- 对照本仓库的 OpenAI 写法，理解 Claude 的 tool use 消息格式差异（`tool_use` / `tool_result` block）。
- 顺带认清 tool 的三种类型：function tool（本仓库用的）、custom tool（自由文本）、built-in tool（平台执行，含 MCP 入口）。

### 阶段 3 · 多轮循环的多种模式
不加新工具，用现有两个工具把这几种循环各跑一遍，观察消息流：
- **Drill-down（钻取）**：概览 → 推理选一个 → 看细节（本仓库现有例子）。
- **Fan-out then aggregate（发散再收敛）**：一次比较多城市，再汇总排名。
- **Retry / self-correction（自我纠错）**：工具报错 → 错误回给模型 → 换参数重试。
- **ReAct（Reason + Act）**：模型先「想」再「做」，每步根据结果调整。理解本仓库循环就是简化版 ReAct。

### 阶段 4 · 设计一个「业务问答」型 agent（多工具、固定 scope）
- 选一个 scope，例如「某公司的数据与业务问答」。
- 设计 5~10 个工具（原子能力），如：查订单、查库存、查客户信息、查财务指标、检索文档等。
- 重点练：如何用 `description` 划清工具边界、scope 外问题如何礼貌拒答、多个工具如何协作。

### 阶段 5 · 设计一个「反复思考、多轮执行」的 agent
- 在阶段 4 基础上，让 agent 具备规划能力：先拆解任务 → 逐步执行 → 观察结果 → 修正计划。
- 引入 planning / scratchpad（思考草稿）、失败重试、终止条件（`max_steps` 之外的收敛判断）。
- 目标：面对一个模糊的大问题，agent 能自己决定调哪些工具、调几轮，直到给出可靠答案。

## 说明

- 这里的工具调用不是通过 LangChain 实现，而是直接使用 OpenAI 的 function calling / tool calling API。
- 消息流是标准的 agent 循环：`system → user → assistant(tool_calls) → tool → assistant(tool_calls) → tool → ... → assistant(最终回答)`。
  模型可以调用工具零次、一次或多次（例如“对比北京和成都”会一次触发两个 `lookup_weather` 调用）。
- LangGraph 版本用一个 `agent` 节点（调用 LLM）和一个 `tools` 节点（执行工具）构成循环：
  `agent → 有 tool_calls? → tools → 回到 agent`，没有 tool_calls 时结束。这与直接版是同一套逻辑，只是换成图来编排。
- 真实联网查询使用 Open-Meteo 的公开 API，所以不需要额外购买天气数据服务。
