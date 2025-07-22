# llm_client_redis

LLM通过 REDIS 的客户端

## 功能

### python api 调用

`llm_client_redis.llm_client.py`

```python

from llm_client_redis import LLMClientRedis
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

llm_client_redis: LLMClientRedis = LLMClientRedis(llm_json_path="../config/llm_resources.json", 
                                                  config_path="../config/config.ini")

model: str = "home_qwen3:32b"

messages: [BaseMessage] = [SystemMessage("你是一个好助手"), HumanMessage("你好")]

data = llm_client_redis.request(messages=messages, model=model)

print(data)

```

### cmd 调用

```shell
chat-session
```