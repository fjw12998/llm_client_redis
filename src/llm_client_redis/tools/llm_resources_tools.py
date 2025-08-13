import json
import logging

from langchain_core.messages import BaseMessage

class LLMResourcesTools:

    def __init__(self, json_path: str):

        with open(json_path, 'r', encoding='utf-8') as f:
            self.llm_dic: dict = json.load(f)

        logging.info(f'Loaded {len(self.llm_dic)} LLM resources from {json_path}')
        logging.info(f'Available LLM resources: {self.llm_dic.keys()}')

    @staticmethod
    def format_messages_to_text(base_messages: list[BaseMessage]) -> str:
        """
        自定义消息转文本函数（按模型需求调整格式）
        :return:
        """
        return "\n".join([f"'{msg.type}': '{msg.content}'" for msg in base_messages]).strip()

    def list_llm_def(self):
        """
        列出所有 LLM 资源
        :return: LLM 资源 key 列表
        """
        return [key for key in self.llm_dic.keys()]

    def is_model_available(self, llm_key: str) -> bool:
        """
        判断是否存在指定 LLM 资源
        :param llm_key: LLM 资源 key
        :return: True or False
        """
        return llm_key in self.llm_dic
