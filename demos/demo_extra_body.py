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


def test_request_stream_v31(llm_client_redis: LLMClientRedis):
    """
    资料: https://support.huaweicloud.com/usermanual-maas-modelarts/maas-modelarts-0083.html
    我查了上面的资料, DeepSeek-R1 与 DeepSeek-R1-0528 不支持关闭思考模式; DeepSeek-V3.1 与 DeepSeek-V3.2 默认关闭思考模式, 但可以打开
    
    模型名称    thinking.type默认值    thinking.type支持的取值
    
    DeepSeek-R1    enabled    enabled
    
    DeepSeek-R1-0528    enabled    enabled
    
    DeepSeek-V3.1    disabled    enabled/disabled
    
    DeepSeek-V3.2    disabled    enabled/disabled

    """
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

def test_request_stream_v3(llm_client_redis: LLMClientRedis):
    """
    测试 deepseek-v3 是否能实现继写，即支持 continue_final_message 与 add_generation_prompt 参数
    """
    model: str = "huawei_deepseek_v3_64k"
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
    test_request_stream_v31(llm_client_redis)
    test_request_stream_v3(llm_client_redis)
