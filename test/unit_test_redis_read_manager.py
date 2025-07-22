from configparser import ConfigParser
from unittest.mock import Mock, MagicMock, patch
import os
import pytest
import redis
from langchain_core.messages import BaseMessage, HumanMessage
from redis import RedisError

from llm_repeator_redis import RedisManager


@pytest.fixture
def configparser_mock():
    # 创建ConfigParser的MagicMock实例
    configparser = MagicMock(spec=ConfigParser)
    # 配置当访问'redis_server' section时返回预设的字典
    redis_server_section = {
        'host': 'redis02.home',
        'port': 16379,
        'password_env_var_name': 'REDIS02_AUTHENTICATION',
        'db': 2,
        'request_stream_name': 'request_stream',
        'answer_map_name': 'answer_map'

    }

    # 配置当访问'redis_arch' section时返回预设的字典
    redis_arch_section = {
        'redis_arch_enable': 'true',
        'redis_arch_host': 'redis02.home',
        'redis_arch_port': 16379,
        'redis_arch_password_env_var_name': 'REDIS02_AUTHENTICATION',
        'redis_arch_db': 3,
        'redis_arch_data_stream_name': 'arch_stream'
    }

    local_llm_section = {
        "local_llm_id": "home_deepseek-r1:8b-llama-distill-fp16"
    }

    # 模拟 __getitem__ 方法以返回不同的部分
    def getitem_side_effect(section):
        if section == 'redis_server':
            return redis_server_section
        elif section == 'redis_arch':
            return redis_arch_section
        elif section == 'local_llm':
            return local_llm_section
        else:
            raise KeyError(f"No section '{section}' found")

    configparser.__getitem__.side_effect = getitem_side_effect
    return configparser

@pytest.fixture
def mock_redis():
    return Mock(spec=redis.Redis)


def test_redis_read_manager(configparser_mock):
    redis_read_manager = RedisManager(configparser=configparser_mock)

    # 验证环境变量存在且不为空
    password_env_var_name = redis_read_manager.password_env_var_name
    assert password_env_var_name == 'REDIS02_AUTHENTICATION'
    assert os.getenv(password_env_var_name) is not None and os.getenv(password_env_var_name) != ''

    # 验证配置项的值
    assert redis_read_manager.host == 'redis02.home'
    assert redis_read_manager.port == 16379
    assert redis_read_manager.db == 2

def test_push_request(configparser_mock):
    """
    要调试请求时，
    :param configparser_mock:
    :return:
    """
    redis_read_manager = RedisManager(configparser=configparser_mock)

    messages: [BaseMessage] = [
        HumanMessage(content="你好！")
    ]

    for i in range(12):
        seq: int = redis_read_manager.push_request(stream_name=configparser_mock['redis_server']['request_stream_name'],
                                                   messages=messages,
                                                   model="home_deepseek-r1:8b-llama-distill-fp16",
                                                   enable_arch=True)

        assert seq is not None


def test_pop_request(configparser_mock):
    redis_read_manager = RedisManager(configparser=configparser_mock)

    data: dict = redis_read_manager.pop_request(configparser_mock['redis_server']['request_stream_name'])

    messages: [BaseMessage] = data['messages']

    assert data is not None
    assert data['seq'] is not None
    assert data['seq'] > 0
    # print(messages)
    assert messages[0]['content'] == "你好！"




