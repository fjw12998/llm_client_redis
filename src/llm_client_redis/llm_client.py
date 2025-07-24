import logging
import time
import os
import redis

from configparser import ConfigParser
from typing import Any, Generator, List
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from .tools import LLMResourcesTools
from .tools import LLMRedisManager
from pathlib import Path

# 模块级别的日志配置
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

class LLMClientRedis:
    """
    使用 Redis 实现的 LLM 请求器
    """
    def __init__(self, llm_json_path: str = None, config_path: str = None):

        # 设置默认路径
        home = Path.home()
        default_config_dir = home / ".llm_client_redis" / "config"
        default_config_dir.mkdir(parents=True, exist_ok=True)

        if config_path is None:
            config_path = default_config_dir / "config.ini"
        if llm_json_path is None:
            llm_json_path = default_config_dir / "llm_resources.json"

        # 确保路径是字符串
        config_path = str(config_path)
        llm_json_path = str(llm_json_path)

        # 创建 config.ini 模板（如果不存在）
        if not os.path.exists(config_path):
            self._create_config_ini(config_path)

        # 创建 llm_resources.json 模板（如果不存在）
        if not os.path.exists(llm_json_path):
            self._create_llm_resources_json(llm_json_path)

        configparser: ConfigParser = ConfigParser()
        configparser.read(config_path, encoding="utf-8")

        self.redis_manager = LLMRedisManager(configparser=configparser)
        self.llm_resources_tools = LLMResourcesTools(json_path=llm_json_path)
        self.stream_name = configparser['redis_server']['request_stream_name']
        self.answer_map_name = configparser['redis_server']['answer_map_name']
        self.chunk_stream_prefix = configparser['redis_server']['chunk_stream_prefix']
        self.reasoning_stream_prefix = configparser['redis_server']['reasoning_stream_prefix']

    @staticmethod
    def _create_llm_resources_json(path: str):
        import json
        template = {
          "deepseek_r1": {
            "model": "deepseek-reasoner",
            "version": "R1",
            "base_url": "https://api.deepseek.com",
            "type": "BaseChatOpenAI",
            "provider": "langchain-deepseek",
            "env_api_key_name": "DEEPSEEK_API_KEY",
            "response_type": "deepseek-reasoner",
            "description": "DeepSeek R1 模型 LangChain 接口"
          },
          "deepseek_v3": {
            "model": "deepseek-chat",
            "version": "V3",
            "base_url": "https://api.deepseek.com",
            "type": "BaseChatOpenAI",
            "provider": "langchain-deepseek",
            "env_api_key_name": "DEEPSEEK_API_KEY",
            "response_type": "deepseek-chat",
            "description": "DeepSeek V3 模型 LangChain 接口"
          },
          "huawei_deepseek_r1_32k": {
            "model": "DeepSeek-R1",
            "version": "R1",
            "base_url": "https://maas-cn-southwest-2.modelarts-maas.com/deepseek-r1/v1",
            "type": "BaseChatOpenAI",
            "provider": "langchain-deepseek",
            "env_api_key_name": "HUAWEI_MODEL_ART_API_KEY",
            "response_type": "deepseek-reasoner",
            "description": "华为云的 DeepSeek R1 32K 模型 LangChain DeepSeek 接口"
          },
          "huawei_deepseek_v3_32k": {
            "model": "DeepSeek-V3",
            "version": "V1",
            "base_url": "https://maas-cn-southwest-2.modelarts-maas.com/deepseek-v3/v1",
            "type": "BaseChatOpenAI",
            "provider": "langchain-deepseek",
            "env_api_key_name": "HUAWEI_MODEL_ART_API_KEY",
            "response_type": "deepseek-chat",
            "description": "华为云的 DeepSeek V3 32K 模型 LangChain DeepSeek 接口"
          },
          "huawei_DeepSeek-R1-32K-0528": {
            "model": "deepseek-r1-250528",
            "version": "R1",
            "base_url": "https://api.modelarts-maas.com/v1",
            "type": "BaseChatOpenAI",
            "provider": "langchain-deepseek",
            "env_api_key_name": "HUAWEI_MODEL_ART_API_KEY",
            "response_type": "deepseek-reasoner",
            "description": "华为云的 DeepSeek-R1-32K-0528 模型 LangChain DeepSeek 接口"
          },
          "huawei_qwen3-32b": {
            "model": "qwen3-32b",
            "version": "V1",
            "base_url": "https://api.modelarts-maas.com/v1",
            "type": "BaseChatOpenAI",
            "provider": "langchain-deepseek",
            "env_api_key_name": "HUAWEI_MODEL_ART_API_KEY",
            "response_type": "deepseek-chat",
            "description": "华为云的 qwen3-32b 模型 LangChain DeepSeek 接口"
          },
          "huawei_qwen3-235b-a22b": {
            "model": "qwen3-235b-a22b",
            "version": "V1",
            "base_url": "https://api.modelarts-maas.com/v1",
            "type": "BaseChatOpenAI",
            "provider": "langchain-deepseek",
            "env_api_key_name": "HUAWEI_MODEL_ART_API_KEY",
            "response_type": "deepseek-chat",
            "description": "华为云的 qwen3-235b-a22b 模型 LangChain DeepSeek 接口"
          },
          "home_deepseek-r1:32b": {
            "model": "deepseek-r1:32b",
            "version": "32b",
            "base_url": "https://localhost:11434",
            "type": "BaseLLM",
            "provider": "langchain-ollama",
            "env_api_key_name": None,
            "response_type": "deepseek-reasoner",
            "description": "本地 DeepSeek R1 32b 模型 LangChain 接口"
          },
          "home_qwen3:32b": {
            "model": "qwen3:32b",
            "version": "32b",
            "base_url": "https://localhost:11434",
            "type": "BaseLLM",
            "provider": "langchain-ollama",
            "env_api_key_name": None,
            "response_type": "deepseek-reasoner",
            "description": "本地 qwen3:32b 模型 LangChain 接口"
          }
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(template, f, indent=4)
        logger.info(f"Created default llm_resources.json at {path}")

    @staticmethod
    def _create_config_ini(path: str):
        template = """# config.ini

[redis_server]
# Redis 服务器的主机地址 006.jq48.cn
host = redis03.home

# Redis 服务器的端口号
port = 26379

# Redis 密码的环境变量名称（通过环境变量读取密码）
password_env_var_name = REDIS03_AUTHENTICATION

# Redis 数据库编号
db = 2

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
redis_arch_enable = true

# Redis 归档服务器的主机地址 006.jq48.cn
redis_arch_host = redis03.home

# Redis 归档服务器的端口号
redis_arch_port = 26379

# Redis 归档服务器密码的环境变量名称（通过环境变量读取密码）
redis_arch_password_env_var_name = REDIS03_AUTHENTICATION

# Redis 归档服务器使用的数据库编号
redis_arch_db = 3

# Redis 归档数据流名称
redis_arch_data_stream_name = arch_stream

[local_llm]
local_llm_id = home_qwen3:32b
"""
        with open(path, "w", encoding="utf-8") as _f:
            _f.write(template)
        logger.info(f"Created default config.ini at {path}")


    def request(self, messages: List[BaseMessage],
                model: str,
                block_time = 20 * 60,
                internal: int = 1,
                enable_arch: bool = True) -> {}:
        """
        将请求消息推送到 Redis 队列中，原样返回答复
        :param messages: 请求消息列表
        :param model: 使用的模型
        :param block_time: 阻塞时间，单位为秒,默认为5分钟
        :param internal: 请求答案的时间间隔，单位为秒，默认为1秒
        :param enable_arch: 是否启用归档，默认为 True
        :return: 请求序列号，请求序号是用于获取响应的
        """
        action_type: str = "generate"

        # 获取当前时间
        current_time = time.time()

        if not self.llm_resources_tools.is_model_available(model):
            logger.error(f"model {model} is not available")
            raise Exception(f"model {model} is not available")

        seq: int = self.redis_manager.push_request(stream_name=self.stream_name,
                                                   messages=messages,
                                                   model=model,
                                                   action_type=action_type,
                                                   enable_arch=enable_arch)

        logger.info(f"push request to redis using model: {model} with action_type: {action_type}, get answer seq: {seq}")

        # 当总时间超过 block_time 跳出循环
        while time.time() - current_time < block_time:

            # 获取请求的答案
            answer = self.redis_manager.pop_response(seq=seq)
            if answer is not None:
                logger.debug(f"seq: {seq} get answer: {answer}")
                return answer
            else:
                time.sleep(internal)
                logger.debug(f"seq: {seq} get answer is None, sleep {internal} seconds")
        logger.error(f"seq: {seq} get answer timeout")
        return None

    def request_stream(self, messages: list[BaseMessage],
                       model: str,
                       block_time=20 * 60,
                       internal: float = 0.02,
                       enable_arch: bool = True) -> Generator[Any, Any, None]:

        action_type: str = "stream"
        current_time = time.time()

        if not self.llm_resources_tools.is_model_available(model):
            logger.error(f"model {model} is not available")
            raise Exception(f"model {model} is not available")

        seq: int = self.redis_manager.push_request(
            stream_name=self.stream_name,
            messages=messages,
            model=model,
            action_type=action_type,
            enable_arch=enable_arch
        )

        logger.info(f"Pushed request to Redis using model: {model}, action_type: {action_type}, seq: {seq}")

        # 是否进行思考过程
        is_reasoning: bool = True

        # 是否首次思考
        is_first_reasoning: bool = False

        while time.time() - current_time < block_time:
            try:

                if is_reasoning:
                    # 尝试获取原因分析
                    chunk_data = self.redis_manager.pop_stream_chunk(seq=seq, chunk_stream_prefix=self.reasoning_stream_prefix)

                    # 原因是直接打印，而不是返回结果
                    if chunk_data and is_first_reasoning == False:
                        is_first_reasoning = True
                        print("<think>")
                        print(chunk_data.decode('utf-8'), end="", flush=True)
                        continue
                    elif chunk_data:
                        print(chunk_data.decode('utf-8'), end="", flush=True)
                        continue

                logger.debug(f"Fetching chunk data from Redis with seq: {seq}, prefix: {self.chunk_stream_prefix}")
                chunk_data = self.redis_manager.pop_stream_chunk(seq=seq, chunk_stream_prefix=self.chunk_stream_prefix)

                if not chunk_data:

                    # 如果数据工作已经结束
                    if self.redis_manager.is_finished_stream(seq=seq, chunk_stream_prefix=self.chunk_stream_prefix):

                        logger.info(f"seq: {seq} stream is finished")

                        self.redis_manager.rem_finish_stream(seq=seq, chunk_stream_prefix=self.chunk_stream_prefix)
                        return None

                    else:
                        logger.debug(f"seq: {seq} no chunk data received, retrying in {internal} seconds")
                        time.sleep(internal)
                        continue

                # 原因为空，且
                if is_reasoning and is_first_reasoning:
                    is_reasoning = False
                    print("\n</think>")

                logger.debug(f"seq: {seq} received chunk data: {chunk_data}")

                yield chunk_data.decode('utf-8')

            except redis.exceptions.RedisError as e:
                logger.error(f"Redis error occurred while processing chunk data for seq {seq}: {e}")
                break
            except Exception as e:
                logger.error(f"Unexpected error occurred while processing chunk data for seq {seq}: {e}")
                break

        logger.info(f"Finished processing request stream for seq: {seq}")
        return None

    def request_messages(self, messages: List[BaseMessage],
                         model: str,
                         block_time =20 * 60,
                         internal: int = 1,
                         enable_arch: bool = True) -> {}:
        """
        将请求消息推送到 Redis 队列中，使用 config.ini 中配置的 response_type 来处理响应，再返回结果
        :param messages: 请求消息列表
        :param model: 使用的模型
        :param block_time: 阻塞时间，单位为秒,默认为5分钟
        :param internal: 请求答案的时间间隔，单位为秒，默认为1秒
        :param enable_arch: 是否启用归档，默认为 True
        :return: 请求序列号，请求序号是用于获取响应的
        """
        answer = self.request(messages=messages, model=model, block_time=block_time, internal=internal, enable_arch=enable_arch)

        return answer

    def request_str_human(self, system: str,
                          human: str,
                          model: str,
                          block_time =20 * 60,
                          internal: int = 1,
                          enable_arch: bool = True) -> {}:
        """
        将请求消息推送到 Redis 队列中，使用 config.ini 中配置的 response_type 来处理响应，再返回结果
        :param system: 提示词
        :param human: 问题
        :param model: 使用的模型
        :param block_time: 阻塞时间，单位为秒,默认为5分钟
        :param internal: 请求答案的时间间隔，单位为秒，默认为1秒
        :param enable_arch: 是否启用归档，默认为 True
        :return:
        """
        messages: List[BaseMessage] = [SystemMessage(system), HumanMessage(human)]

        return self.request_messages(messages=messages, model=model, block_time=block_time, internal=internal, enable_arch=enable_arch)

    def request_file_human(self, system_file_path: str, human: str, model: str, block_time =20 * 60, internal: int = 1, enable_arch: bool = True) -> {}:
        """
        将请求消息推送到 Redis 队列中，使用 config.ini 中配置的 response_type 来处理响应，再返回结果
        :param system_file_path: 提示词文件路径
        :param human: 问题
        :param model: 使用的模型
        :param block_time: 阻塞时间，单位为秒,默认为5分钟
        :param internal: 请求答案的时间间隔，单位为秒，默认为1秒
        :param enable_arch: 是否启用归档，默认为 True
        :return: 请求序列号，请求序号是用于获取响应的
        """
        with open(system_file_path, 'r', encoding='utf-8') as f:
            prompt: str = f.read()

        messages: List[BaseMessage] = [SystemMessage(prompt), HumanMessage(human)]

        return self.request_messages(messages=messages, model=model, block_time=block_time, internal=internal, enable_arch=enable_arch)
