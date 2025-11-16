from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from llm_client_redis import LLMClientRedis
import logging
from typing import List
from ftplib import print_line

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def test_request(llm_client_redis: LLMClientRedis):

    model: str = "huawei_deepseek_v3.1"

    messages: List[BaseMessage] = [HumanMessage("写个100字的故事"), AIMessage("不好")]

    data = llm_client_redis.request(messages=messages, 
                                    model=model, 
                                    max_tokens=150, 
                                    continue_final_message=True, 
                                    add_generation_prompt=False)

    logging.info(data)


def test_request_stream(llm_client_redis: LLMClientRedis):
    model: str = "huawei_deepseek_v3.1"
    messages: List[BaseMessage] = [HumanMessage("写个100字的故事"), AIMessage("不好")]
    
    result: str = ""
    
    for data in llm_client_redis.request_stream(messages=messages, 
                                    model=model, 
                                    max_tokens=150, 
                                    continue_final_message=True, 
                                    add_generation_prompt=False):
        print(data, flush=True, end="")
        result += data
    
    logging.info(result)

if __name__ == "__main__":
    llm_client_redis:LLMClientRedis = LLMClientRedis(llm_json_path="../config/llm_resources.json", config_path="../config/config.ini")
    test_request(llm_client_redis)
    test_request_stream(llm_client_redis)
