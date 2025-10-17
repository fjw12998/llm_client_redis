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

### 4. llm-watch-dirs 命令说明

`llm-watch-dirs` 是一个目录监控工具，用于监控指定目录中的文件变化，并将文件内容发送到大语言模型进行处理。

#### 命令选项

- `-p`, `--prompt_paths`: 指定监控的目录，多个目录使用英文逗号分隔。与 `-w` 选项不可同时使用。
- `-o`, `--output_path`: 指定输出子目录名称，默认为 `results`
- `-i`, `--interval`: 指定监控的间隔时间，默认为 60 秒
- `-r`, `--random-start`: 是否随机开始，随机的时间是由参数 [i](file://d:\git-huaweicloud\llm_client_redis\config\config.ini) 决定，默认为 `False`
- `-w`, `--watch-dir`: 指定监控目录，此目录下的所有子目录将会被纳入监控。与 `-p` 选项不可共存。注意目录的名称为需要使用的模型名称加上_数字，例如: `/path/to/model_1,/path/to/model_2`

#### 使用方法

```bash
# 监控指定的多个目录
llm-watch-dirs -p /path/to/model_1,/path/to/model_2 -o results -i 60

# 监控目录下的所有子目录
llm-watch-dirs -w /path/to/watch_dir -o results -i 30

# 随机开始监控
llm-watch-dirs -p /path/to/model_1 -r
```

### 工作原理

该工具会定期扫描监控目录中的文件，查找指定后缀名的文件（默认为 [.md](file://d:\git-huaweicloud\llm_client_redis\README.md), [.txt](file://d:\git-huaweicloud\llm_client_redis\requirements.txt), `.pro`, `.prompt`），将文件内容发送到对应的大语言模型进行处理，并将结果保存到每个目录的 `results` 子目录中。

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
