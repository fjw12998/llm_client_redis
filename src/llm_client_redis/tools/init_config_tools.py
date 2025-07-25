import json

import logging

class InitConfigTools:

    @staticmethod
    def create_llm_resources_json(path: str):
        template = {
            "deepseek_r1": {
                "model": "deepseek-reasoner",
                "version": "R1",
                "base_url": "https://api.deepseek.com",
                "type": "BaseChatOpenAI",
                "provider": "langchain-deepseek",
                "env_api_key_name": "DEEPSEEK_API_KEY",
                "response_type": "deepseek-reasoner",
                "description": "DeepSeek R1 \u6a21\u578b LangChain \u63a5\u53e3"
            },
            "deepseek_v3": {
                "model": "deepseek-chat",
                "version": "V3",
                "base_url": "https://api.deepseek.com",
                "type": "BaseChatOpenAI",
                "provider": "langchain-deepseek",
                "env_api_key_name": "DEEPSEEK_API_KEY",
                "response_type": "deepseek-chat",
                "description": "DeepSeek V3 \u6a21\u578b LangChain \u63a5\u53e3"
            },
            "huawei_deepseek_r1_32k": {
                "model": "DeepSeek-R1",
                "version": "R1",
                "base_url": "https://maas-cn-southwest-2.modelarts-maas.com/deepseek-r1/v1",
                "type": "BaseChatOpenAI",
                "provider": "langchain-deepseek",
                "env_api_key_name": "HUAWEI_MODEL_ART_API_KEY",
                "response_type": "deepseek-reasoner",
                "description": "\u534e\u4e3a\u4e91\u7684 DeepSeek R1 32K \u6a21\u578b LangChain DeepSeek \u63a5\u53e3"
            },
            "huawei_deepseek_v3_32k": {
                "model": "DeepSeek-V3",
                "version": "V1",
                "base_url": "https://maas-cn-southwest-2.modelarts-maas.com/deepseek-v3/v1",
                "type": "BaseChatOpenAI",
                "provider": "langchain-deepseek",
                "env_api_key_name": "HUAWEI_MODEL_ART_API_KEY",
                "response_type": "deepseek-chat",
                "description": "\u534e\u4e3a\u4e91\u7684 DeepSeek V3 32K \u6a21\u578b LangChain DeepSeek \u63a5\u53e3"
            },
            "huawei_DeepSeek-R1-32K-0528": {
                "model": "deepseek-r1-250528",
                "version": "R1",
                "base_url": "https://api.modelarts-maas.com/v1",
                "type": "BaseChatOpenAI",
                "provider": "langchain-deepseek",
                "env_api_key_name": "HUAWEI_MODEL_ART_API_KEY",
                "response_type": "deepseek-reasoner",
                "description": "\u534e\u4e3a\u4e91\u7684 DeepSeek-R1-32K-0528 \u6a21\u578b LangChain DeepSeek \u63a5\u53e3"
            },
            "huawei_qwen3-32b": {
                "model": "qwen3-32b",
                "version": "V1",
                "base_url": "https://api.modelarts-maas.com/v1",
                "type": "BaseChatOpenAI",
                "provider": "langchain-deepseek",
                "env_api_key_name": "HUAWEI_MODEL_ART_API_KEY",
                "response_type": "deepseek-chat",
                "description": "\u534e\u4e3a\u4e91\u7684 qwen3-32b \u6a21\u578b LangChain DeepSeek \u63a5\u53e3"
            },
            "huawei_qwen3-235b-a22b": {
                "model": "qwen3-235b-a22b",
                "version": "V1",
                "base_url": "https://api.modelarts-maas.com/v1",
                "type": "BaseChatOpenAI",
                "provider": "langchain-deepseek",
                "env_api_key_name": "HUAWEI_MODEL_ART_API_KEY",
                "response_type": "deepseek-chat",
                "description": "\u534e\u4e3a\u4e91\u7684 qwen3-235b-a22b \u6a21\u578b LangChain DeepSeek \u63a5\u53e3"
            },
            "home_deepseek-r1:32b": {
                "model": "deepseek-r1:32b",
                "version": "32b",
                "base_url": "https://localhost:11434",
                "type": "BaseLLM",
                "provider": "langchain-ollama",
                "env_api_key_name": None,
                "response_type": "deepseek-reasoner",
                "description": "\u672c\u5730 DeepSeek R1 32b \u6a21\u578b LangChain \u63a5\u53e3"
            },
            "home_qwen3:32b": {
                "model": "qwen3:32b",
                "version": "32b",
                "base_url": "https://localhost:11434",
                "type": "BaseLLM",
                "provider": "langchain-ollama",
                "env_api_key_name": None,
                "response_type": "deepseek-reasoner",
                "description": "\u672c\u5730 qwen3:32b \u6a21\u578b LangChain \u63a5\u53e3"
            }
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(template, f, indent=4)
        logging.info(f"Created default llm_resources.json at {path}")

    @staticmethod
    def create_config_ini(path: str):
        template: str = """# config.ini v0.1.0

[redis_server]
# Redis 服务器的主机地址
host = localhost

# Redis 服务器的端口号
port = 6379

# Redis 密码的环境变量名称（通过环境变量读取密码）
password_env_var_name = REDIS_AUTHENTICATION

# Redis 数据库编号
db = 0

# 用于请求的 Redis 数据流名称
request_stream_name = request_stream

# 用于响应的 Redis 数据配对名称
answer_map_name = answer_map

# 用于响应的 Redis 数据流名称前辍，前辍包含:号，加上序号则表示响应流 list 的名称
chunk_stream_prefix = chunk_stream:

# 用于深入分析的 Redis 数据流名称前辍，前辍包含:号，加上序号则表示响应流 list 的名称
reasoning_stream_prefix = reasoning_stream:

[logging]
# 日志级别，可选值有 DEBUG, INFO, WARNING, ERROR, CRITICAL
level = INFO


[redis_arch]
# 是否启用 Redis 归档功能，true 表示启用，false 表示禁用
redis_arch_enable = false

# Redis 归档服务器的主机地址
redis_arch_host = localhost

# Redis 归档服务器的端口号
redis_arch_port = 6379

# Redis 归档服务器密码的环境变量名称（通过环境变量读取密码）
redis_arch_password_env_var_name = REDIS_AUTHENTICATION

# Redis 归档服务器使用的数据库编号
redis_arch_db = 1

# Redis 归档数据流名称
redis_arch_data_stream_name = arch_stream

[local_llm]
local_llm_id = home_qwen3:32b
"""

        with open(path, "w", encoding="utf-8") as _f:
            _f.write(template)
        logging.info(f"Created default config.ini at {path}")

    @staticmethod
    def refresh_llm_resources():
        """
        TODO 实现从 redis 服务器刷新客户段配置文件的功能，需要后期实现，服务器端也需要相关的实现
        :return:
        """
        pass
