import os
from pathlib import Path

import pytest

from src.llm_client_redis import LLMClientRedis


@pytest.fixture
def llm_client_redis() -> LLMClientRedis:
    return LLMClientRedis()


def test_request(llm_client_redis: LLMClientRedis):

    assert llm_client_redis is not None

    home = Path.home()

    assert os.path.exists(f'{home}/.llm_client_redis/config/config.ini')
    assert os.path.exists(f'{home}/.llm_client_redis/config/llm_resources.json')





