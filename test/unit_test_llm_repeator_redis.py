import pytest
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

from llm_client_redis import LLMClientRedis


@pytest.fixture
def llm_client_redis() -> LLMClientRedis:
    return LLMClientRedis(llm_json_path="../config/llm_resources.json", config_path="../config/config.ini")


def test_request(llm_client_redis: LLMClientRedis):

    model: str = "home_qwen3:32b"

    messages: [BaseMessage] = [SystemMessage("你是一个好助手"), HumanMessage("你好")]

    data = llm_client_redis.request(messages=messages, model=model)

    print(data)

    assert data is not None




