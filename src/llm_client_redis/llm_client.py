import logging
import time
import os
import redis

from configparser import ConfigParser
from typing import Any, Generator, List, Optional
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from .tools import LLMResourcesTools
from .tools import LLMRedisManager
from .tools import InitConfigTools
from pathlib import Path

class LLMClientRedis:
    """
    使用 Redis 实现的 LLM 请求器
    """
    def __init__(self, llm_json_path: Optional[str] = None, config_path: Optional[str] = None):
        """
        初始化 LLMClientRedis 对象
        :param llm_json_path: llm 资源文件路径
        :param config_path: 配置文件路径
        """

        # 设置默认路径
        home = Path.home()
        default_config_dir = home / ".llm_client_redis" / "config"
        default_config_dir.mkdir(parents=True, exist_ok=True)

        if not config_path:
            config_path = default_config_dir / "config.ini"
        if not llm_json_path:
            llm_json_path = default_config_dir / "llm_resources.json"

        # 确保路径是字符串
        config_path = str(config_path)
        llm_json_path = str(llm_json_path)

        # 创建 config.ini 模板（如果不存在）
        if not os.path.exists(config_path):
            InitConfigTools.create_config_ini(config_path)
            logging.info(f"Programme create a config file, you must set it correctly. path: {config_path}")

        # 创建 llm_resources.json 模板（如果不存在）
        if not os.path.exists(llm_json_path):
            InitConfigTools.create_llm_resources_json(llm_json_path)
            logging.info(f"Programme create a llm_resources.json file, you can add other llm to it. path: {llm_json_path}")

        configparser: ConfigParser = ConfigParser()
        configparser.read(config_path, encoding="utf-8")

        self.redis_manager = LLMRedisManager(configparser=configparser)
        self.llm_resources_tools = LLMResourcesTools(json_path=llm_json_path)
        self.stream_name = configparser['redis_server']['request_stream_name']
        self.answer_map_name = configparser['redis_server']['answer_map_name']
        self.chunk_stream_prefix = configparser['redis_server']['chunk_stream_prefix']
        self.reasoning_stream_prefix = configparser['redis_server']['reasoning_stream_prefix']


    def request(self, messages: List[BaseMessage],
                model: str,
                block_time = 20 * 60,
                internal: float = 0.5,
                enable_arch: bool = True) -> Optional[dict]:
        """
        将请求消息推送到 Redis 队列中，原样返回答复
        :param messages: 请求消息列表
        :param model: 使用的模型
        :param block_time: 阻塞时间，单位为秒,默认为20分钟
        :param internal: 请求答案的时间间隔,单位为秒,默认为0.5秒
        :param enable_arch: 是否启用归档，默认为 True
        :return: 请求序列号，请求序号是用于获取响应的
        """
        action_type: str = "generate"

        # 获取当前时间
        current_time = time.time()

        if not self.llm_resources_tools.is_model_available(model):
            logging.error(f"model {model} is not available")
            raise Exception(f"model {model} is not available")

        seq: int = self.redis_manager.push_request(stream_name=self.stream_name,
                                                   messages=messages,
                                                   model=model,
                                                   action_type=action_type,
                                                   enable_arch=enable_arch)

        logging.info(f"push request to redis using model: {model} with action_type: {action_type}, get answer seq: {seq}")

        # 当总时间超过 block_time 跳出循环
        while time.time() - current_time < block_time:

            # 获取请求的答案
            answer = self.redis_manager.pop_response(seq=seq)
            if answer is not None:
                logging.debug(f"seq: {seq} get answer: {answer}")
                return answer
            else:
                time.sleep(internal)
                logging.debug(f"seq: {seq} get answer is None, sleep {internal} seconds")
        logging.error(f"seq: {seq} get answer timeout")
        return None

    def request_stream(self, messages: list[BaseMessage],
                       model: str,
                       block_time=20 * 60,
                       internal: float = 0.02,
                       enable_arch: bool = True) -> Optional[Generator[Any, Any, None]]:

        action_type: str = "stream"
        current_time = time.time()

        if not self.llm_resources_tools.is_model_available(model):
            logging.error(f"model {model} is not available")
            raise Exception(f"model {model} is not available")

        seq: int = self.redis_manager.push_request(
            stream_name=self.stream_name,
            messages=messages,
            model=model,
            action_type=action_type,
            enable_arch=enable_arch
        )

        logging.info(f"Pushed request to Redis using model: {model}, action_type: {action_type}, seq: {seq}")

        # 是否进行思考过程
        is_reasoning: bool = True

        # 是否首次思考
        is_first_reasoning: bool = False

        while time.time() - current_time < block_time:
            try:

                if is_reasoning:
                    # 尝试获取原因分析
                    chunk_data: bytes = self.redis_manager.pop_stream_chunk(seq=seq, chunk_stream_prefix=self.reasoning_stream_prefix)

                    # 原因是直接打印，而不是返回结果
                    if chunk_data and is_first_reasoning == False:
                        is_first_reasoning = True
                        print("<think>")
                        print(chunk_data.decode('utf-8'), end="", flush=True)
                        continue
                    elif chunk_data:
                        print(chunk_data.decode('utf-8'), end="", flush=True)
                        continue

                logging.debug(f"Fetching chunk data from Redis with seq: {seq}, prefix: {self.chunk_stream_prefix}")
                chunk_data = self.redis_manager.pop_stream_chunk(seq=seq, chunk_stream_prefix=self.chunk_stream_prefix)

                if not chunk_data:

                    # 如果数据工作已经结束
                    if self.redis_manager.is_finished_stream(seq=seq, chunk_stream_prefix=self.chunk_stream_prefix):

                        logging.info(f"seq: {seq} stream is finished")

                        # 是否存在一个可能，就是流已标记结束，但刚好还有数据在 stream_chunk 的队列中没有补取出？
                        end_chunk_data = self.redis_manager.pop_stream_chunk(seq=seq,
                                                                             chunk_stream_prefix=self.chunk_stream_prefix)

                        if end_chunk_data:
                            logging.warning(f"seq: {seq} stream is finished, but there is still data in the stream_chunk queue, data: {end_chunk_data}")
                            chunk_data = end_chunk_data
                        else:
                            logging.debug(f"seq: {seq} stream is finished, no data in the stream_chunk queue")

                            self.redis_manager.rem_finish_stream(seq=seq, chunk_stream_prefix=self.chunk_stream_prefix)
                            return None

                    else:
                        logging.debug(f"seq: {seq} no chunk data received, retrying in {internal} seconds")
                        time.sleep(internal)
                        continue

                # 原因为空，且
                if is_reasoning and is_first_reasoning:
                    is_reasoning = False
                    print("\n</think>")

                logging.debug(f"seq: {seq} received chunk data: {chunk_data}")

                yield chunk_data.decode('utf-8')

            except redis.exceptions.RedisError as e:
                logging.error(f"Redis error occurred while processing chunk data for seq {seq}: {e}")
                break
            except Exception as e:
                logging.error(f"Unexpected error occurred while processing chunk data for seq {seq}: {e}")
                break

        logging.info(f"Finished processing request stream for seq: {seq}")
        return None

    def request_messages(self, messages: List[BaseMessage],
                         model: str,
                         block_time =20 * 60,
                         internal: float = 0.5,
                         enable_arch: bool = True) -> {}:
        """
        将请求消息推送到 Redis 队列中，使用 config.ini 中配置的 response_type 来处理响应，再返回结果
        :param messages: 请求消息列表
        :param model: 使用的模型
        :param block_time: 阻塞时间，单位为秒,默认为20分钟
        :param internal: 请求答案的时间间隔，单位为秒，默认为0.5秒
        :param enable_arch: 是否启用归档，默认为 True
        :return: 请求序列号，请求序号是用于获取响应的
        """
        answer = self.request(messages=messages,
                              model=model,
                              block_time=block_time,
                              internal=internal,
                              enable_arch=enable_arch)

        return answer

    def request_str_human(self, system: str,
                          human: str,
                          model: str,
                          block_time =20 * 60,
                          internal: float = 0.5,
                          enable_arch: bool = True) -> {}:
        """
        将请求消息推送到 Redis 队列中，使用 config.ini 中配置的 response_type 来处理响应，再返回结果
        :param system: 提示词
        :param human: 问题
        :param model: 使用的模型
        :param block_time: 阻塞时间，单位为秒,默认为20分钟
        :param internal: 请求答案的时间间隔，单位为秒，默认为0.5秒
        :param enable_arch: 是否启用归档，默认为 True
        :return:
        """
        messages: List[BaseMessage] = [SystemMessage(system), HumanMessage(human)]

        return self.request_messages(messages=messages,
                                     model=model,
                                     block_time=block_time,
                                     internal=internal,
                                     enable_arch=enable_arch)

    def request_file_human(self,
                           system_file_path: str,
                           human: str,
                           model: str,
                           block_time = 20 * 60,
                           internal: float = 0.5,
                           enable_arch: bool = True) -> {}:
        """
        将请求消息推送到 Redis 队列中，使用 config.ini 中配置的 response_type 来处理响应，再返回结果
        :param system_file_path: 提示词文件路径
        :param human: 问题
        :param model: 使用的模型
        :param block_time: 阻塞时间，单位为秒,默认为20分钟
        :param internal: 请求答案的时间间隔，单位为秒，默认为0.5秒
        :param enable_arch: 是否启用归档，默认为 True
        :return: 请求序列号，请求序号是用于获取响应的
        """
        with open(system_file_path, 'r', encoding='utf-8') as f:
            prompt: str = f.read()

        messages: List[BaseMessage] = [SystemMessage(prompt), HumanMessage(human)]

        return self.request_messages(messages=messages,
                                     model=model,
                                     block_time=block_time,
                                     internal=internal,
                                     enable_arch=enable_arch)

def init():
    LLMClientRedis()
    logging.info("The init() function has been called")