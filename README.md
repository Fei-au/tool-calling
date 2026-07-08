# Tool Calling Weather Demo

这个仓库展示了两种方式实现“网上查询天气”的 tool calling 学习示例：

1. 直接使用 OpenAI 的 tool calling API
2. 使用 LangGraph 编排一个联网查询天气的流程

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

## 说明

- 这里的工具调用不是通过 LangChain 实现，而是直接使用 OpenAI 的 function calling / tool calling API。
- LangGraph 负责把“解析用户问题 → 调用天气工具 → 生成自然语言回答”串成一个工作流。
- 真实联网查询使用 Open-Meteo 的公开 API，所以不需要额外购买天气数据服务。
