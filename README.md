# llm_client_redis

## 介绍
整合多种llm 的api接入，使用 redis 作为消息队列，实现多客户端并发调用 llm 服务。本项目是调用部分，还有另一个项目专门用于接收 redis 消息，
实现与 llm 的通信，并将返回结果给 redis，

## 软件架构
软件架构说明


## 安装教程

### 1. 使用 `PyPI` 安装
```commandline
pip install llm_client_redis
```

### 2. 项目安装
```commandline
pip install -r requirements.txt
```

### 3. 完成安装后进行配置文件初始化

执行如下的命令，可以对 `llm-client-redis` 生成初始的配置文件

```commandline
llm-client-init
```

路径在 `~/.llm-client-redis/config/` 下，分别生成 `config.ini` 和 `llm_resources.json` 文件

* `config.ini` 文件用于配置 `redis` 的连接信息
* `llm_resources.json` 文件用于配置 `llm` 的信息，包括 `llm` 的名称，需要与服务器端一致

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

进入命令行模式，实现调用

### 3. restful api 调用

执行以下命令，打开 web resultful api 服务

```shell
uvicorn src.llm_client_redis.llm_restful_client_main:app --reload
```

* 通过调用 url `http://localhost:8000/models` 可以获取所有可用的 `llm` 模型
* 通过调用 url `http://localhost:8000/demo.json` 实现流程的 demo
* 流式访问 linux 版
```bash
curl -X POST http://localhost:8000/stream -H "Content-Type: application/json" -d '{"message": "你好，世界！"}'
```
* 流式访问 windows 版
```bash
curl -X POST http://localhost:8000/stream -H "Content-Type: application/json" -d "{\"message\": \"你好，世界！\"}"
```

## 参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request



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
