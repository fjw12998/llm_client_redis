import json
import logging
import os
import time
from configparser import ConfigParser
from datetime import datetime, timezone
from typing import Optional

import redis
from langchain_core.messages import BaseMessage
from redis import RedisError
from redis.exceptions import TimeoutError


class LLMRedisManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LLMRedisManager, cls).__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self, configparser: Optional[ConfigParser] = None):
        """
        从 config.ini 文件中读取 Redis 服务器的配置信息
        """
        if self.__initialized:
            return
        self.__initialized = True

        if configparser is None:
            configparser = ConfigParser()

            config_file_path: str = os.path.join(os.path.dirname(__file__), './config', 'config.ini')

            if not os.path.exists(config_file_path) or not configparser.read(config_file_path, encoding='utf-8'):
                raise FileNotFoundError(f'配置文件不存在或无法读取 {config_file_path}')

        # 读取 Redis 服务器配置
        self.host = configparser['redis_server']['host']
        self.port = int(configparser['redis_server']['port'])
        self.password_env_var_name = configparser['redis_server']['password_env_var_name']

        if self.password_env_var_name not in os.environ:
            raise ValueError(f'环境变量 {self.password_env_var_name} 未设置，请确保已正确配置环境变量')

        self.db = int(configparser['redis_server']['db'])

        self.request_stream_name = configparser['redis_server']['request_stream_name']

        self.answer_map_name = configparser['redis_server']['answer_map_name']

        logging.info('RedisManager initialized with host: %s, '
                     'port: %s, '
                     'password_env_var_name: %s, '
                     'db: %s, '
                     'request_stream_name: %s',
                     self.host, self.port, self.password_env_var_name, self.db, self.request_stream_name)

        self._connection_pool = redis.ConnectionPool(
            host=self.host,
            port=self.port,
            password=os.getenv(self.password_env_var_name),
            db=self.db
        )

        # 读取 Redis 存档配置
        self.redis_arch_enable: bool = str.lower(configparser['redis_arch'].get('redis_arch_enable', "true")) == "true"

        logging.info('RedisManager initialized with redis_arch_enable: %s', self.redis_arch_enable)

        if self.redis_arch_enable:
            self.redis_arch_host: str = configparser['redis_arch']['redis_arch_host']
            self.redis_arch_port: int = int(configparser['redis_arch']['redis_arch_port'])
            self.redis_arch_password_env_var_name: str = configparser['redis_arch']['redis_arch_password_env_var_name']
            self.redis_arch_db: int = int(configparser['redis_arch']['redis_arch_db'])
            self.redis_arch_data_stream_name: str = configparser['redis_arch']['redis_arch_data_stream_name']

            logging.info('RedisManager initialized with redis_arch_host: %s, redis_arch_port: %s, '
                         'redis_arch_password_env_var_name: %s, redis_arch_db: %s, redis_arch_data_stream_name: %s',
                         self.redis_arch_host,
                         self.redis_arch_port,
                         self.redis_arch_password_env_var_name,
                         self.redis_arch_db,
                         self.redis_arch_data_stream_name)

            self._arch_connection_pool = redis.ConnectionPool(
                host=self.redis_arch_host,
                port=self.redis_arch_port,
                password=os.getenv(self.redis_arch_password_env_var_name),
                db=self.redis_arch_db
            )

    def get_redis_connection(self) -> redis.Redis:
        """
        获取 Redis 连接
        :return: redis.Redis
        """
        try:
            return redis.Redis(connection_pool=self._connection_pool)
        except redis.RedisError as e:
            logging.error("Failed to get Redis connection: %s", e)
            raise

    def get_redis_arch_connection(self) -> redis.Redis:
        """
        获取 Redis 存档连接
        :return:
        """
        try:
            return redis.Redis(connection_pool=self._arch_connection_pool)
        except redis.RedisError as e:
            logging.error("Failed to get Redis arch connection: %s", e)
            raise

    def push_request(self, stream_name: str,
                     messages: list[BaseMessage],
                     model: str,
                     action_type: str = "generate",
                     enable_arch: bool = False,
                     retry: int = 6,
                     retry_internal: int = 20,
                     **kwargs) -> int:
        """
        添加请求
        :param stream_name: 流名称
        :param messages: 请求消息列表
        :param model: 模型名称
        :param action_type: 请求类型，默认为 "generate", 可选 "stream", "agenerate", "astream"
        :param enable_arch: 是否启用存档，默认：False
        :param retry: 重试次数, 默认为 6 次
        :param retry_internal: 重试间隔时间，单位为秒, 默认为 20 秒
        :param kwargs: 其他参数
        :return: 请求序列号，请求序号是用于获取响应的
        """
        retry_count: int = 0

        if str.lower(action_type) not in ['generate', 'stream', 'agenerate', 'astream']:
            logging.error("parameter action_type must be one of ['generate', 'stream', 'agenerate', 'astream'], but got %s", action_type)
            raise Exception("parameter action_type must be one of ['generate', 'stream', 'agenerate', 'astream'], but got %s", action_type)
        
        while retry_count < retry:
        
            if retry_count > 0:
                logging.warning(f"正在进行重试操作{retry_count}次")

            try:
                with self.get_redis_connection() as conn:

                    seq: int = conn.incrby(f"request:{stream_name}:count")

                    request: {} = {'seq': seq,
                                'model': model,
                                'action_type': action_type,
                                'messages': [msg.model_dump() for msg in messages],
                                'enable_arch': enable_arch,
                                'llm_kwargs': kwargs}

                    data: str = json.dumps(request)

                    conn.rpush(stream_name, data)

                    return seq
            except Exception as e:
                logging.error("Failed to add request: %s", e)
                if retry_count < retry:
                    retry_count += 1
                    logging.warning(f"重试次数: {retry_count}, 重试间隔: {retry_internal} 秒")
                    time.sleep(retry_internal)
                else:
                    raise RuntimeError(f"Failed to push request to Redis: {e}") from e

    def pop_request(self,
                    stream_name: str,
                    retry: int = 6,
                    retry_internal: int = 20) -> Optional[dict]:
        """
        获取待处理请求
        :param stream_name: 流名称
        :param retry: 重试次数, 默认为 6 次
        :param retry_internal: 重试间隔时间，单位为秒, 默认为 20 秒

        :return: 待处理请求数据
        """
        retry_count: int = 0

        while retry_count < retry:
        
            if retry_count > 0:
                logging.warning(f"正在进行重试操作{retry_count}次")

            try:
                with self.get_redis_connection() as conn:

                    data: str = conn.lpop(name=stream_name)

                    if data:
                        return json.loads(data)
                    else:
                        return None
            except TimeoutError as e:
                logging.error("Failed to get pending requests: %s", e)
                if retry_count < retry:
                    retry_count += 1
                    logging.warning(f"重试次数: {retry_count}, 重试间隔: {retry_internal} 秒")
                    time.sleep(retry_internal)
                else:
                    raise RuntimeError(f"Failed to pop request from Redis: {e}") from e

    def pop_response(self, seq: int,
                     retry: int = 6,
                     retry_internal: int = 20) -> Optional[dict]:
        """
        获取响应结果
        :param seq: 指定序号
        :param retry: 重试次数, 默认为 6 次
        :param retry_internal: 重试间隔时间，单位为秒, 默认为 20 秒
        :return:
        """
        retry_count: int = 0

        while retry_count < retry:
        
            if retry_count > 0:
                logging.warning(f"正在进行重试操作{retry_count}次")

            with self.get_redis_connection() as conn:
                try:
                    data: str = conn.hget(name=self.answer_map_name, key=str(seq))
                    if data:
                        return json.loads(data)
                    else:
                        return None
                except TimeoutError as e:
                    logging.error("Failed to get response for seq %s: %s", seq, e)

                    if retry_count < retry:
                        retry_count += 1
                        logging.warning(f"重试次数: {retry_count}, 重试间隔: {retry_internal} 秒")
                        time.sleep(retry_internal)
                    else:
                        raise RuntimeError(f"Failed to get connection from Redis: {e}") from e
                finally:
                    conn.hdel(self.answer_map_name, str(seq))


    def save_response(self, seq: int,
                      model: str,
                      response: [],
                      retry: int = 6,
                      retry_internal: int = 20) -> None:
        """
        保存响应结果
        :param seq: 请求序列号
        :param model: 模型名称
        :param response: 响应内容
        :param retry: 重试次数, 默认为 6 次
        :param retry_internal: 重试间隔时间，单位为秒, 默认为 20 秒
        """
        retry_count: int = 0

        while retry_count < retry:

            if retry_count > 0:
                logging.warning(f"正在进行重试操作{retry_count}次")

            try:
                with self.get_redis_connection() as conn:
                    conn.hset(name=f"{self.answer_map_name}", key=str(seq), value=json.dumps({
                        "seq": seq,
                        "model": model,
                        "timestamp": str(datetime.now(timezone.utc)),
                        "answer": response
                    }))

            except Exception as e:
                logging.error("Failed to save response for serial %s: %s", seq, e)
                if retry_count < retry:
                    retry_count += 1
                    logging.warning(f"重试次数: {retry_count}, 重试间隔: {retry_internal} 秒")
                    time.sleep(retry_internal)
                else:
                    raise RuntimeError(f"Failed to save response to Redis: {e}") from e

    def stream_chunk(self, seq: int,
                     chunk_text: str,
                     chunk_stream_prefix: str,
                     retry: int = 6,
                     retry_internal: int = 20) -> None:
        f"""
        流式输出到 redis，将 chunk 保存到 {chunk_stream_prefix}{seq} 的列表中
        :param seq: 请求序列号
        :param chunk_text: chunk 文本
        :param chunk_stream_prefix: chunk 流前缀
        :param retry: 重试次数, 默认为 6 次
        :param retry_internal: 重试间隔时间，单位为秒, 默认为 20 秒
        :return: 
        """

        retry_count: int = 0

        while retry_count < retry:
            try:
                with self.get_redis_connection() as conn:
                    conn.lpush(f"{chunk_stream_prefix}{seq}", chunk_text)

                    return
            except Exception as e:
                logging.error("Failed to stream chunk to redis: %s", e)
                if retry_count < retry:
                    retry_count += 1
                    logging.warning(f"重试次数: {retry_count}, 重试间隔: {retry_internal} 秒")
                    time.sleep(retry_internal)
                else:
                    raise RuntimeError(f"Failed to get stream from Redis: {e}") from e

    def pop_stream_chunk(self, seq: int,
                         chunk_stream_prefix: str,
                         retry: int = 6,
                         retry_internal: int = 20) -> Optional[bytes]:
        f"""
        从 redis 中获取流式数据
        如果 ${chunk_stream_prefix}{seq} 的列表不存在时，返回空串
        :param seq: 流数据的序列号
        :param chunk_stream_prefix: 流数据的前缀
        :param retry: 重试次数，默认为6次
        :param retry_internal: 重试间隔时间，默认为20秒
        
        :return: 获取到的流数据，如果列表不存在则返回空串
        """
        retry_count: int = 0

        while retry_count < retry:
            try:
                with self.get_redis_connection() as conn:
                    return conn.rpop(f"{chunk_stream_prefix}{seq}")
            except Exception as e:
                logging.error("Failed to pop stream chunk from redis: %s", e)
                if retry_count < retry:
                    retry_count += 1
                    logging.warning(f"重试次数: {retry_count}, 重试间隔: {retry_internal} 秒")
                    time.sleep(retry_internal)
                else:
                    raise RuntimeError(f"Failed to pop from Redis: {e}") from e

        return None

    def finish_stream(self, seq: int,
                      chunk_stream_prefix: str,
                      retry: int = 6,
                      retry_internal: int = 20) -> None:
        """
        在redis 中，对 stream 设置完成状态，用于告诉获取器，该流的获取已结束

        完成状态被放在 名为 ${chunk_stream_prefix} 的集合中，其值为序号

        :param seq: 请求序列号
        :param chunk_stream_prefix: chunk 流前缀
        :param retry: 重试次数, 默认为 6 次
        :param retry_internal: 重试间隔时间，单位为秒, 默认为 20 秒
        :return:
        """
        retry_count: int = 0

        while retry_count < retry:
            try:
                with self.get_redis_connection() as conn:
                    conn.sadd(f"{chunk_stream_prefix}", seq)

                    return
            except Exception as e:

                logging.error("Failed to save finish status to redis: %s", e)
                if retry_count < retry:
                    retry_count += 1
                    logging.warning(f"重试次数: {retry_count}, 重试间隔: {retry_internal} 秒")
                    time.sleep(retry_internal)
                else:
                    raise RuntimeError(f"Failed set finished status to Redis: {e}") from e

    def is_finished_stream(self, seq: int, chunk_stream_prefix: str,
                          retry: int = 6,
                          retry_internal: int = 20) -> Optional[bool]:
        """
        判断流是否已经完成

        :param seq: 请求序列号
        :param chunk_stream_prefix: chunk 流前缀
        :param retry: 重试次数, 默认为 6 次
        :param retry_internal: 重试间隔时间，单位为秒, 默认为 20 秒
        :return: True 表示流已经完成，False 表示流还未完成
        """
        retry_count: int = 0

        while retry_count < retry:
            try:
                with self.get_redis_connection() as conn:
                    return conn.sismember(f"{chunk_stream_prefix}", str(seq)) == 1
            except Exception as e:

                logging.error("Failed to check finish status from redis: %s", e)
                if retry_count < retry:
                    retry_count += 1
                    logging.warning(f"重试次数: {retry_count}, 重试间隔: {retry_internal} 秒")
                    time.sleep(retry_internal)
                else:
                    raise RuntimeError(f"Failed to check finish status from Redis: {e}") from e

        return None


    def rem_finish_stream(self, seq: int, chunk_stream_prefix: str,
                          retry: int = 6,
                          retry_internal: int = 20) -> None:
        """
        移除完成状态的标记，表示该次访问完全终结

        :param seq: 请求序列号
        :param chunk_stream_prefix: chunk 流前缀
        :param retry: 重试次数, 默认为 6 次
        :param retry_internal: 重试间隔时间，单位为秒, 默认为 20 秒
        :return:
        """
        retry_count: int = 0

        while retry_count < retry:
            try:
                with self.get_redis_connection() as conn:
                    conn.srem(f"{chunk_stream_prefix}", seq)

                    return
            except Exception as e:

                logging.error("Failed to remove finish status from redis: %s", e)
                if retry_count < retry:
                    retry_count += 1
                    logging.warning(f"重试次数: {retry_count}, 重试间隔: {retry_internal} 秒")
                    time.sleep(retry_internal)
                else:
                    raise RuntimeError(f"Failed to remove finished status from Redis: {e}") from e

    def save_to_arch_redis(self,
                           seq: int,
                           model: str,
                           request_json: {},
                           answer_text: str,
                           retry: int = 6,
                           retry_internal: int = 20) -> None:
        """
        将输入消息和输出结果以 JSON 的形式保存到 redis 服务器中，为归档做准备。如果在 config.ini 配置 redis_arch_enable = false时，
        则不进行任何操作
        :param seq: 请求序列号
        :param model: 模型名称
        :param request_json: 请求内容
        :param answer_text: 响应内容
        :param retry: 重试次数, 默认为 6 次
        :param retry_internal: 重试间隔时间，单位为秒, 默认为 20 秒
        """

        if not self.redis_arch_enable:
            logging.debug(f"redis_arch_enable is false, no need to save to arch redis")
            return

        description:str = ""

        # 重试次数
        retry_count: int = 0

        # 构建完整的归档记录（添加 UTC 时间戳）
        record = {
            "seq": seq,
            "model": model,
            "description": description,
            "request": request_json,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "answer": answer_text
        }

        while retry_count < retry:

            try:
                with self.get_redis_arch_connection() as conn:
                    # 将记录写入 Redis
                    conn.rpush(self.redis_arch_data_stream_name, json.dumps(record))
                    logging.info(f"Record for serial {seq} saved to Redis")

            except RedisError as e:
                logging.error("Redis 操作失败: %s", e)

                if retry_count < retry:
                    retry_count += 1
                    logging.warning(f"重试次数: {retry_count}, 重试间隔: {retry_internal} 秒")
                    time.sleep(retry_internal)
                else:
                    raise RedisError(f"Failed to save to Redis: {e}") from e
            except Exception as e:
                logging.error("保存到 Redis 失败: %s", e)

                if retry_count < retry:
                    retry_count += 1
                    logging.warning(f"重试次数: {retry_count}, 重试间隔: {retry_internal} 秒")
                    time.sleep(retry_internal)
                else:
                    raise RuntimeError(f"Failed to save arch data to Redis: {e}") from e

    def read_latest_from_arch_redis(self, count: int = -1,
                                    retry: int = 6,
                                    retry_internal: int = 20) -> Optional[list]:
        """
        从 Redis 流中读取最新数据
        :param count: 获取数量
        :param retry: 重试次数, 默认为 6 次
        :param retry_internal: 重试间隔时间，单位为秒, 默认为 20 秒
        :return: 读取到的数据
        """

        # 重试次数
        retry_count: int = 0

        while retry_count < retry:
            try:
                with self.get_redis_arch_connection() as conn:
                    # 获取 Redis 中最新的数据 '>'
                    data: list = conn.lrange(name=self.redis_arch_data_stream_name, start=0, end=count)

                    if data is None or len(data) == 0:
                        return []

                    conn.ltrim(name=self.redis_arch_data_stream_name, start=len(data), end=-1)

                    return [json.loads(item) for item in data]
            except Exception as e:
                logging.error("Failed to read latest data from Redis: %s", e)

                if retry_count < retry:
                    retry_count += 1
                    logging.warning(f"重试次数: {retry_count}, 重试间隔: {retry_internal} 秒")
                    time.sleep(retry_internal)
                else:
                    raise RedisError(f"Failed to read last data from Redis: {e}") from e

        return None

