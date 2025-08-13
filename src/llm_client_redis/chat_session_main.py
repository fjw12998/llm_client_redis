from configparser import ConfigParser
from pathlib import Path

from llm_client_redis import LLMClientRedis
from llm_client_redis.cmd_chat import ChatSession
from llm_client_redis.cmd_chat.cmd_templates import SendTemplateCommand
from llm_client_redis.cmd_chat.cmd_templates import ShowTextBlockCommand
from llm_client_redis.cmd_chat.cmd_templates import TextBlockCommand


def main():

    home_path: Path = Path.home()

    # 初始化配置文件
    LLMClientRedis()

    config_path: str = f"{str(home_path)}/.llm_client_redis/config/config.ini"

    configparser: ConfigParser = ConfigParser()
    configparser.read(config_path, encoding="utf-8")

    model: str = configparser['local_llm']['local_llm_id']

    chat = ChatSession(model=model,
                       llm_json_path=f"{str(home_path)}/.llm_client_redis/config/llm_resources.json",
                       config_path=config_path,
                       max_history=8)  # 保存最近4轮对话

    chat.command_registry.register(SendTemplateCommand())
    chat.command_registry.register(TextBlockCommand())
    chat.command_registry.register(ShowTextBlockCommand())

    chat.start()
    pass

if __name__ == '__main__':
    main()