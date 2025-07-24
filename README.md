# llm_client_redis



## 功能




# llm_client_redis

## 介绍
整合多种llm 的api接入，使用 redis 作为消息队列，实现多客户端并发调用 llm 服务。本项目是调用部分，还有另一个项目专门用于接收 redis 消息，
实现与 llm 的通信，并将返回结果给 redis，

## 软件架构
软件架构说明


## 安装教程

1.  xxxx
2.  xxxx
3.  xxxx

## 使用说明

### 1. python api 调用

`llm_client_redis.llm_client.py`

一次获取所有回答内容，等待出现相应的时间会较长

```python

from src.llm_client_redis import LLMClientRedis
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from typing import List

llm_client_redis: LLMClientRedis = LLMClientRedis(llm_json_path="../config/llm_resources.json",
                                                  config_path="../config/config.ini")

model: str = "home_qwen3:32b"

messages: List[BaseMessage] = [SystemMessage("你是一个好助手"), HumanMessage("你好")]

data = llm_client_redis.request(messages=messages, model=model)

print(data)

```

### 2. cmd 调用

```shell
chat-session
```

## 参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request


## 特技

1.  使用 Readme\_XXX.md 来支持不同的语言，例如 Readme\_en.md, Readme\_zh.md
2.  Gitee 官方博客 [blog.gitee.com](https://blog.gitee.com)
3.  你可以 [https://gitee.com/explore](https://gitee.com/explore) 这个地址来了解 Gitee 上的优秀开源项目
4.  [GVP](https://gitee.com/gvp) 全称是 Gitee 最有价值开源项目，是综合评定出的优秀开源项目
5.  Gitee 官方提供的使用手册 [https://gitee.com/help](https://gitee.com/help)
6.  Gitee 封面人物是一档用来展示 Gitee 会员风采的栏目 [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)
