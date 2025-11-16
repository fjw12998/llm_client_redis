from typing import List

import pytest
import logging
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from llm_client_redis import LLMClientRedis


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

@pytest.fixture
def llm_client_redis() -> LLMClientRedis:
    return LLMClientRedis(llm_json_path="../config/llm_resources.json", config_path="../config/config.ini")


def test_request(llm_client_redis: LLMClientRedis):

    model: str = "home_qwen3:32b"

    messages: List[BaseMessage] = [SystemMessage("你是一个好助手"), HumanMessage("你好")]

    data = llm_client_redis.request(messages=messages, model=model)

    logging.info(data)

    assert data is not None


def test_push_request(llm_client_redis: LLMClientRedis):

    model: str = "huawei_deepseek_v3.1"

    messages: List[BaseMessage] = [HumanMessage("写个100字的故事"), AIMessage("不好")]

    result: str = ""

    for data in llm_client_redis.request(messages=messages, 
                                    model=model, 
                                    max_tokens=150, 
                                    continue_final_message=True, 
                                    add_generation_prompt=False):
        print(data, flush=True, end="")
        result += data

    logging.info(result)

    assert result is not None